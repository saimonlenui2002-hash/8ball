package com.example.pooltrajectory

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.min

/**
 * ML 6.2 is deliberately a guard/refinement layer around the frozen 6.1 core.
 *
 * The field test that motivated 6.2 showed that 6.1 already fixed the dangerous
 * 180-degree aim reversals and often gets the ghost/contact physics right. Rather
 * than replacing that geometry, 6.2 keeps 6.1 authoritative and only intervenes
 * when diagnostics expose one of the remaining failure modes:
 *  - a short/weak guide suddenly wins over the previously stable main guide;
 *  - rayFallback is treated as confidently as a native-ghost solution;
 *  - the rendered coloured branch starts a few pixels after the visible native
 *    white post-contact branch instead of continuing it cleanly.
 *
 * This wrapper is intentionally conservative: it never synthesizes a new target or
 * collision normal. If a 6.2 guard is not clearly triggered, the 6.1 result passes
 * through unchanged.
 */
class GeometryFirstAnalyzerV62(context: Context) {
    private val core = GeometryFirstAnalyzerV61(context)

    private data class ParsedDiag(
        val source: String? = null,
        val guideLen: Double? = null,
        val inSupport: Double? = null,
        val targetQuality: Double? = null,
        val radiusRatio: Double? = null,
        val alignment: Double? = null,
        val ghostScore: Double? = null
    )

    private data class StableFrame(
        val result: AnalysisResult,
        val atMs: Long,
        val guideLen: Double?,
        val inSupport: Double?,
        val source: String?
    )

    private var stable: StableFrame? = null

    fun analyze(bitmap: Bitmap): AnalysisResult {
        val now = System.currentTimeMillis()
        val base = core.analyze(bitmap)
        val parsed = parseDiagnostics(base.diagnostics)

        // 6.1 already owns the physical geometry. 6.2 first decides whether this
        // frame is trustworthy enough to replace the last accepted frame.
        val guarded = guardShortGuideJump(base, parsed, now)
        val fallbackChecked = calibrateFallback(guarded, parsed, now)
        val branchSnapped = snapBranchStarts(bitmap, fallbackChecked)

        val finalDiag = appendDiag(
            branchSnapped.diagnostics,
            "v62=guarded",
            "v62Source=${parsed.source ?: "none"}",
            "v62BranchSnap=${if (branchSnapped.segments != fallbackChecked.segments) 1 else 0}"
        )
        val finalResult = branchSnapped.copy(diagnostics = finalDiag)

        val retainedByGuard = finalResult.diagnostics.contains("v62GuideGate=hold") ||
            finalResult.diagnostics.contains("v62Fallback=retained")
        if (!retainedByGuard && isStableCandidate(finalResult, parsed)) {
            stable = StableFrame(finalResult, now, parsed.guideLen, parsed.inSupport, parsed.source)
        } else if (stable != null && now - stable!!.atMs > STABLE_TTL_MS) {
            stable = null
        }
        return finalResult
    }

    private fun guardShortGuideJump(base: AnalysisResult, d: ParsedDiag, now: Long): AnalysisResult {
        val previous = stable ?: return base
        if (now - previous.atMs > GUIDE_HOLD_MS) return base

        val cue = base.cueBall ?: return base
        val prevCue = previous.result.cueBall ?: return base
        val aim = base.aimDirection?.normalized() ?: return base
        val prevAim = previous.result.aimDirection?.normalized() ?: return base
        if (aim.length() < 0.5 || prevAim.length() < 0.5) return base

        val sameCue = (cue.center - prevCue.center).length() <= max(10.0, cue.radius * 0.72)
        if (!sameCue) return base

        val jump = angleDeg(aim, prevAim)
        val currentLen = d.guideLen ?: return base
        val previousLen = previous.guideLen ?: return base
        val currentSupport = d.inSupport ?: 0.0
        val previousSupport = previous.inSupport ?: 0.0

        // Trigger only on a large discontinuity AND evidence that the newly chosen
        // Hough segment is materially worse/shorter. Legitimate slow aiming is left
        // untouched, and even a fast aim move passes if the new guide remains strong.
        val muchShorter = currentLen < previousLen * 0.60
        val muchWeaker = currentSupport + 0.20 < previousSupport
        if (jump < GUIDE_JUMP_DEG || (!muchShorter && !muchWeaker)) return base

        val retained = previous.result.copy(
            frameWidth = base.frameWidth,
            frameHeight = base.frameHeight,
            playRect = base.playRect ?: previous.result.playRect,
            confidence = min(previous.result.confidence, 82),
            status = "ML 6.2: удержана проверенная штатная линия",
            diagnostics = appendDiag(
                base.diagnostics,
                "v62GuideGate=hold",
                "v62AimJump=${fmt(jump)}",
                "v62GuideRatio=${fmt(currentLen / max(1.0, previousLen))}",
                "v62SupportDelta=${fmt(currentSupport - previousSupport)}"
            )
        )
        return retained
    }

    private fun calibrateFallback(base: AnalysisResult, d: ParsedDiag, now: Long): AnalysisResult {
        if (d.source != "rayFallback") return base

        val weakReasons = mutableListOf<String>()
        d.inSupport?.let { if (it < 0.22) weakReasons += "support" }
        d.targetQuality?.let { if (it < 0.50) weakReasons += "targetQ" }
        d.radiusRatio?.let { if (it !in 0.70..1.30) weakReasons += "ratio" }
        d.alignment?.let { if (it < 0.20) weakReasons += "align" }

        val previous = stable
        val fallbackJump = previous != null &&
            previous.source == "nativeGhost" &&
            now - previous.atMs <= FALLBACK_RETAIN_MS &&
            sameCueAndAim(base, previous.result, 3.0) &&
            targetJumpPx(base, previous.result) > max(28.0, (base.cueBall?.radius ?: 12.0) * 2.20)
        if (fallbackJump) weakReasons += "targetJump"

        if (weakReasons.isNotEmpty()) {
            val previous = stable
            val canRetain = previous != null &&
                now - previous.atMs <= FALLBACK_RETAIN_MS &&
                sameCueAndAim(base, previous.result, 4.0)
            if (canRetain) {
                return previous!!.result.copy(
                    frameWidth = base.frameWidth,
                    frameHeight = base.frameHeight,
                    playRect = base.playRect ?: previous.result.playRect,
                    confidence = min(previous.result.confidence, 76),
                    status = "ML 6.2: слабый fallback — сохранена последняя проверенная геометрия",
                    diagnostics = appendDiag(
                        base.diagnostics,
                        "v62Fallback=retained",
                        "v62FallbackWeak=${weakReasons.joinToString(",")}"
                    )
                )
            }

            // Do not draw a confident long prediction from weak fallback evidence.
            // Debug AIM segments are kept so diagnostics still show what was seen.
            val debugOnly = base.segments.filter { it.kind == SegmentKind.AIM }
            return base.copy(
                segments = debugOnly,
                confidence = min(base.confidence, 44),
                status = "ML 6.2: fallback не подтверждён",
                diagnostics = appendDiag(
                    base.diagnostics,
                    "v62Fallback=rejected",
                    "v62FallbackWeak=${weakReasons.joinToString(",")}"
                )
            )
        }

        // Valid fallback remains available, but it can no longer claim native-ghost
        // confidence. This is calibration only; the 6.1 geometry itself is unchanged.
        val confidenceCap = if ((d.ghostScore ?: 0.0) >= 0.68)
            FALLBACK_WITH_GHOST_CONFIDENCE_CAP else FALLBACK_CONFIDENCE_CAP
        return base.copy(
            confidence = min(base.confidence, confidenceCap),
            status = if (base.status.startsWith("ML 6.1:")) {
                base.status.replaceFirst("ML 6.1:", "ML 6.2:")
            } else base.status,
            diagnostics = appendDiag(base.diagnostics, "v62Fallback=accepted")
        )
    }

    private fun snapBranchStarts(bitmap: Bitmap, result: AnalysisResult): AnalysisResult {
        val ghost = result.ghostCueCenter ?: return result
        if (result.segments.none { it.kind == SegmentKind.OBJECT || it.kind == SegmentKind.CUE_AFTER }) return result

        var objectDone = false
        var cueDone = false
        var changed = false
        val targetRadius = result.targetBall?.radius ?: result.cueBall?.radius ?: 16.0
        val cueRadius = result.cueBall?.radius ?: targetRadius

        val adjusted = result.segments.map { seg ->
            val eligible = when (seg.kind) {
                SegmentKind.OBJECT -> !objectDone.also { objectDone = true }
                SegmentKind.CUE_AFTER -> !cueDone.also { cueDone = true }
                else -> false
            }
            if (!eligible) return@map seg

            val dir = (seg.end - seg.start).normalized()
            if (dir.length() < 0.5) return@map seg
            val currentDistance = (seg.start - ghost).dot(dir)
            if (currentDistance <= 0.0) return@map seg

            val expectedFloor = when (seg.kind) {
                SegmentKind.OBJECT -> max(8.0, cueRadius + targetRadius * 0.55)
                SegmentKind.CUE_AFTER -> max(8.0, cueRadius * 1.10)
                else -> 8.0
            }
            val measuredLast = lastNativeWhite(bitmap, ghost, dir, expectedFloor)
            if (measuredLast <= 0.0) return@map seg

            val desired = measuredLast + max(2.0, bitmap.width * 0.0010)
            val maxBackShift = max(10.0, cueRadius * 0.90)
            if (desired >= currentDistance - 1.5 || currentDistance - desired > maxBackShift) return@map seg

            val newStart = ghost + dir * desired
            if (!insideBitmap(newStart, bitmap)) return@map seg
            changed = true
            seg.copy(start = newStart)
        }
        return if (changed) result.copy(segments = adjusted) else result
    }

    private fun lastNativeWhite(bitmap: Bitmap, origin: Vec2, dir: Vec2, startDistance: Double): Double {
        val d = dir.normalized()
        if (d.length() < 0.5) return 0.0
        val perp = Vec2(-d.y, d.x)
        val outer = min(150.0, max(64.0, bitmap.width * 0.060))
        val corridor = max(2.0, bitmap.width * 0.0015).coerceAtMost(4.5)
        var r = startDistance.coerceAtLeast(6.0)
        var last = 0.0
        var run = 0
        var lastRunEnd = 0.0
        while (r <= outer) {
            val hit = whiteAcross(bitmap, origin + d * r, perp, corridor)
            if (hit) {
                run++
                last = r
                if (run >= 2) lastRunEnd = r
            } else {
                run = 0
            }
            r += 1.5
        }
        return if (lastRunEnd > 0.0) lastRunEnd else last
    }

    private fun whiteAcross(bitmap: Bitmap, p: Vec2, perp: Vec2, corridor: Double): Boolean {
        var samples = 0
        var hits = 0
        var o = -corridor
        while (o <= corridor + 1e-6) {
            val q = p + perp * o
            val x = q.x.toInt()
            val y = q.y.toInt()
            if (x in 0 until bitmap.width && y in 0 until bitmap.height) {
                samples++
                if (isAchromaticWhite(bitmap.getPixel(x, y))) hits++
            }
            o += 1.25
        }
        return samples >= 2 && hits >= max(2, (samples * 0.45).toInt())
    }

    private fun isAchromaticWhite(pixel: Int): Boolean {
        val r = Color.red(pixel)
        val g = Color.green(pixel)
        val b = Color.blue(pixel)
        val hi = max(r, max(g, b))
        val lo = min(r, min(g, b))
        if (hi < 155) return false
        // Approximate HSV S<=70 without converting every sampled pixel to HSV.
        val saturation255 = if (hi == 0) 0.0 else (hi - lo) * 255.0 / hi
        return saturation255 <= 74.0
    }

    private fun isStableCandidate(r: AnalysisResult, d: ParsedDiag): Boolean {
        if (r.cueBall == null || r.aimDirection == null) return false
        if (r.segments.none { it.kind == SegmentKind.OBJECT || it.kind == SegmentKind.CUE_AFTER }) return false
        if (r.confidence < 56) return false
        if (d.source == "rayFallback" && r.confidence > FALLBACK_WITH_GHOST_CONFIDENCE_CAP) return false
        return true
    }

    private fun targetJumpPx(a: AnalysisResult, b: AnalysisResult): Double {
        val at = a.targetBall ?: return 0.0
        val bt = b.targetBall ?: return 0.0
        return (at.center - bt.center).length()
    }

    private fun sameCueAndAim(a: AnalysisResult, b: AnalysisResult, maxAngleDeg: Double): Boolean {
        val ac = a.cueBall ?: return false
        val bc = b.cueBall ?: return false
        if ((ac.center - bc.center).length() > max(10.0, ac.radius * 0.75)) return false
        val ad = a.aimDirection?.normalized() ?: return false
        val bd = b.aimDirection?.normalized() ?: return false
        return angleDeg(ad, bd) <= maxAngleDeg
    }

    private fun parseDiagnostics(s: String): ParsedDiag {
        fun number(pattern: Regex): Double? = pattern.find(s)?.groupValues?.getOrNull(1)?.toDoubleOrNull()
        val source = Regex("(?:^|; )source=([^;]+)").find(s)?.groupValues?.getOrNull(1)
        return ParsedDiag(
            source = source,
            guideLen = number(Regex("(?:^|; )guideLen=([-+0-9.]+)")),
            inSupport = number(Regex("(?:^|; )inSupport=([-+0-9.]+)")),
            targetQuality = number(Regex("(?:^|; )target=[^;]*? q=([-+0-9.]+)")),
            radiusRatio = number(Regex("(?:^|; )target=[^;]*? ratio=([-+0-9.]+)")),
            alignment = number(Regex("(?:^|; )target=[^;]*? align=([-+0-9.]+)")),
            ghostScore = number(Regex("(?:^|; )ghostScore=([-+0-9.]+)"))
        )
    }

    private fun appendDiag(base: String, vararg fields: String): String {
        val suffix = fields.filter { it.isNotBlank() }.joinToString("; ")
        if (suffix.isBlank()) return base
        return if (base.isBlank()) suffix else "$base; $suffix"
    }

    private fun insideBitmap(p: Vec2, bitmap: Bitmap): Boolean =
        p.x >= 1.0 && p.x < bitmap.width - 1.0 && p.y >= 1.0 && p.y < bitmap.height - 1.0

    private fun angleDeg(a: Vec2, b: Vec2): Double {
        val dot = a.normalized().dot(b.normalized()).coerceIn(-1.0, 1.0)
        return Math.toDegrees(acos(dot))
    }

    private fun fmt(v: Double) = String.format(java.util.Locale.US, "%.3f", v)

    companion object {
        private const val GUIDE_HOLD_MS = 300L
        private const val GUIDE_JUMP_DEG = 16.0
        private const val FALLBACK_RETAIN_MS = 260L
        private const val STABLE_TTL_MS = 1200L
        private const val FALLBACK_CONFIDENCE_CAP = 72
        private const val FALLBACK_WITH_GHOST_CONFIDENCE_CAP = 82
    }
}
