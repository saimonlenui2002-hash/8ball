package com.example.pooltrajectory

import android.content.Context
import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * 6.0 geometry-first analyzer.
 *
 * The old 4.x/5.0 path tried to discover and classify the two tiny native
 * post-contact branches first. That is fragile: a one-pixel branch error can
 * move a long extrapolated line by many pixels and can also swap object/cue.
 *
 * 6.0 changes the authority chain:
 *   1) detect the long native incoming aim line;
 *   2) detect real ball centres/radii;
 *   3) choose the cue ball from line alignment + achromatic interior;
 *   4) ray-cast the cue-centre path against every inflated target circle;
 *   5) compute the ghost centre and the two equal-mass post-contact vectors;
 *   6) only then use short native white pixels in a narrow cone as an OPTIONAL
 *      sub-degree correction, never as branch identity;
 *   7) continue the corrected rays to effective cushion boundaries inset by
 *      the actual ball radius.
 *
 * Accuracy is preferred over coverage: if cue/target/guide do not agree, no
 * long trajectory is drawn. Cached ball centres may bridge a short Hough miss
 * while the current native aim line is still present; stale aim lines are never
 * reused, so a shot immediately hides the prediction.
 */
class GeometryFirstAnalyzerV6(private val context: Context) {
    private val prefs = context.getSharedPreferences(MainActivity.PREFS, Context.MODE_PRIVATE)

    private data class CircleCandidate(
        val center: Vec2,
        val radius: Double,
        val whiteFraction: Double,
        val edgeCoverage: Double,
        val texture: Double
    ) {
        val quality: Double
            get() = (edgeCoverage * 0.48 + texture * 0.30 + min(1.0, whiteFraction * 1.2) * 0.22)
                .coerceIn(0.0, 1.0)
    }

    private data class Guide(
        val a: Vec2,
        val b: Vec2,
        val length: Double,
        val quality: Double
    )

    private data class CueGuide(
        val cue: CircleCandidate,
        val guide: Guide,
        val direction: Vec2,
        val score: Double
    )

    private data class TargetHit(
        val target: CircleCandidate,
        val t: Double,
        val ghost: Vec2,
        val normal: Vec2,
        val score: Double
    )

    private data class RayEvidence(
        val direction: Vec2,
        val score: Double,
        val lastWhite: Double
    )

    private data class HitRect(
        val point: Vec2,
        val vertical: Boolean,
        val horizontal: Boolean
    )

    private var frameSeq = 0L
    private var cacheFrameWidth = 0
    private var cacheFrameHeight = 0
    private var cacheTableWidth = 0
    private var cacheTableHeight = 0
    private var cachedBalls: List<CircleCandidate> = emptyList()
    private var cacheAtMs = 0L
    private var lastCueCenter: Vec2? = null

    fun analyze(bitmap: Bitmap): AnalysisResult {
        frameSeq++
        val now = System.currentTimeMillis()
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)

        // Keep more pixels than 4.x because centre error dominates long-line error.
        val scale = min(1.0, 1600.0 / max(1, src.cols()).toDouble())
        val work = Mat()
        if (scale < 0.999) {
            Imgproc.resize(src, work, Size(), scale, scale, Imgproc.INTER_AREA)
        } else {
            src.copyTo(work)
        }
        src.release()

        val tableRect = fixedTable(work)
        if (tableRect == null) {
            work.release()
            clearCache()
            return emptyResult(bitmap, "ML 6.0: кадр слишком мал")
        }

        val playRect = PlayRect(
            tableRect.x / scale,
            tableRect.y / scale,
            (tableRect.x + tableRect.width) / scale,
            (tableRect.y + tableRect.height) / scale
        )
        val table = work.submat(tableRect)
        val gray = Mat()
        Imgproc.cvtColor(table, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.medianBlur(gray, gray, 5)
        val hsv = Mat()
        Imgproc.cvtColor(table, hsv, Imgproc.COLOR_RGBA2RGB)
        Imgproc.cvtColor(hsv, hsv, Imgproc.COLOR_RGB2HSV)
        val whiteMask = buildWhiteMask(table)
        val edges = Mat()
        Imgproc.Canny(gray, edges, 48.0, 145.0)

        val guides = detectGuides(whiteMask, table.cols().toDouble())
        if (guides.isEmpty()) {
            gray.release(); hsv.release(); whiteMask.release(); edges.release(); table.release(); work.release()
            clearCache()
            return AnalysisResult(
                bitmap.width, bitmap.height, null, emptyList(), null, emptyList(), playRect,
                18, "ML 6.0: ожидание штатной линии прицеливания",
                diagnostics = "guide=0"
            )
        }

        val compatibleCache = cacheFrameWidth == bitmap.width &&
            cacheFrameHeight == bitmap.height &&
            cacheTableWidth == table.cols() && cacheTableHeight == table.rows() &&
            now - cacheAtMs <= 1100L

        // Heavy full-table circle detection can run every other analyzed frame once
        // a stable cache exists. Guide detection still runs on every analyzed frame.
        val runBallDetection = !compatibleCache || cachedBalls.size < 4 || frameSeq % 2L == 0L
        val freshBalls = if (runBallDetection) detectBalls(gray, hsv, edges, table.cols().toDouble()) else emptyList()
        val balls = mergeBallPool(
            freshBalls,
            if (compatibleCache) cachedBalls else emptyList(),
            table.cols().toDouble()
        )

        if (freshBalls.size >= 3) {
            cachedBalls = mergeBallPool(freshBalls, if (compatibleCache) cachedBalls else emptyList(), table.cols().toDouble())
                .take(28)
            cacheFrameWidth = bitmap.width
            cacheFrameHeight = bitmap.height
            cacheTableWidth = table.cols()
            cacheTableHeight = table.rows()
            cacheAtMs = now
        }

        val cueGuide = chooseCueAndGuide(guides, balls, table.cols().toDouble())
        if (cueGuide == null) {
            val dbg = guides.take(3).map { g ->
                TrajectorySegment(toGlobal(g.a, tableRect, scale), toGlobal(g.b, tableRect, scale), SegmentKind.AIM)
            }
            gray.release(); hsv.release(); whiteMask.release(); edges.release(); table.release(); work.release()
            return AnalysisResult(
                bitmap.width, bitmap.height, null, emptyList(), null,
                if (prefs.getBoolean(MainActivity.KEY_DEBUG, false)) dbg else emptyList(),
                playRect, 30, "ML 6.0: биток не подтверждён",
                diagnostics = "guides=${guides.size}; balls=${balls.size}; cue=0"
            )
        }

        lastCueCenter = cueGuide.cue.center
        val targetHit = chooseTarget(cueGuide.cue, cueGuide.direction, balls, table.cols().toDouble())
        if (targetHit == null) {
            val cueBallGlobal = cueGuide.cue.toBall(tableRect, scale, true)
            val aim = TrajectorySegment(
                cueBallGlobal.center,
                toGlobal(cueGuide.cue.center + cueGuide.direction * (table.cols() * 0.16), tableRect, scale),
                SegmentKind.AIM
            )
            gray.release(); hsv.release(); whiteMask.release(); edges.release(); table.release(); work.release()
            return AnalysisResult(
                bitmap.width, bitmap.height, cueBallGlobal, listOf(cueBallGlobal), null,
                if (prefs.getBoolean(MainActivity.KEY_DEBUG, false)) listOf(aim) else emptyList(),
                playRect, 42, "ML 6.0: цель на луче не подтверждена",
                aimDirection = toGlobalDirection(cueGuide.direction),
                diagnostics = diagnosticBase(guides.size, balls.size, cueGuide, null, null, null)
            )
        }

        // Snap the analytically predicted ghost centre to the visible native ghost
        // ring only when a nearby ring-like response is strong. This fixes visual
        // continuity without letting a random ring choose the physics.
        val ghostVisual = refineGhostCenter(
            whiteMask,
            targetHit.ghost,
            cueGuide.cue.radius,
            table.cols().toDouble()
        )
        val aimFromCentres = (ghostVisual - cueGuide.cue.center).normalized()
        val aimDir = if (aimFromCentres.length() > 0.5 && angleDeg(aimFromCentres, cueGuide.direction) <= 1.8) {
            aimFromCentres
        } else {
            cueGuide.direction
        }

        // Re-solve target with the final incoming direction so the ghost and normal
        // are internally consistent after the optional ring snap.
        val refinedTargetHit = rayCircleHit(
            cueGuide.cue,
            aimDir,
            targetHit.target,
            table.cols().toDouble()
        ) ?: targetHit
        var ghost = refinedTargetHit.ghost
        if ((ghostVisual - ghost).length() <= max(5.0, table.cols() * 0.0065)) ghost = ghostVisual
        var normal = (refinedTargetHit.target.center - ghost).normalized()
        if (normal.length() < 0.5) normal = refinedTargetHit.normal

        val normalSpeed = aimDir.dot(normal).coerceIn(0.0, 1.0)
        val objectPhysics = normal
        val cueVector = aimDir - normal * normalSpeed
        val cueResidual = cueVector.length()
        val cuePhysics = if (cueResidual >= 0.055) cueVector.normalized() else null

        // Native short pixels may correct a direction, but only inside a tight cone
        // around the geometry-predicted branch. This prevents branch swaps entirely.
        val objectNative = refineExpectedBranch(
            whiteMask, ghost, objectPhysics, aimDir,
            table.cols().toDouble(), searchDegrees = 15.0
        )
        val objectDir = objectNative?.direction ?: objectPhysics

        val cueNative = cuePhysics?.let {
            refineExpectedBranch(
                whiteMask, ghost, it, aimDir,
                table.cols().toDouble(), searchDegrees = 18.0
            )
        }
        val cueDir = cueNative?.direction ?: cuePhysics

        // Rendering starts at the far end of the game's own short white branch when
        // visible. Direction and identity remain geometry-derived.
        val objectStart = branchRenderStart(
            whiteMask, ghost, objectDir, objectNative,
            cueGuide.cue.radius, refinedTargetHit.target.radius,
            table.cols().toDouble(), isObject = true
        )
        val cueStart = cueDir?.let {
            branchRenderStart(
                whiteMask, ghost, it, cueNative,
                cueGuide.cue.radius, refinedTargetHit.target.radius,
                table.cols().toDouble(), isObject = false
            )
        }

        val bounces = prefs.getInt(MainActivity.KEY_BOUNCES, 0).coerceIn(0, 2)
        val segmentsLocal = mutableListOf<TrajectorySegment>()
        segmentsLocal += traceToCushions(
            objectStart,
            objectDir,
            localPlayRect(table),
            refinedTargetHit.target.radius,
            SegmentKind.OBJECT,
            bounces
        )
        if (cueDir != null && cueStart != null && cueResidual >= 0.075) {
            segmentsLocal += traceToCushions(
                cueStart,
                cueDir,
                localPlayRect(table),
                cueGuide.cue.radius,
                SegmentKind.CUE_AFTER,
                bounces
            )
        }

        val debugEnabled = prefs.getBoolean(MainActivity.KEY_DEBUG, false)
        if (debugEnabled) {
            // Incoming measured native line, then a short marker from cue centre to
            // analytically chosen ghost centre. These are diagnostics only.
            segmentsLocal += TrajectorySegment(cueGuide.guide.a, cueGuide.guide.b, SegmentKind.AIM)
            segmentsLocal += TrajectorySegment(cueGuide.cue.center, ghost, SegmentKind.AIM)
        }

        val segmentsGlobal = segmentsLocal.map { it.toGlobal(tableRect, scale) }
        val cueGlobal = cueGuide.cue.toBall(tableRect, scale, true)
        val targetGlobal = refinedTargetHit.target.toBall(tableRect, scale, false)
        val debugBalls = listOf(cueGlobal, targetGlobal)
        val ghostGlobal = toGlobal(ghost, tableRect, scale)

        val guideQ = cueGuide.guide.quality
        val cueQ = (cueGuide.cue.whiteFraction * 0.62 + cueGuide.cue.edgeCoverage * 0.38).coerceIn(0.0, 1.0)
        val targetQ = refinedTargetHit.target.quality
        val nativeBonus = (if (objectNative != null) 0.05 else 0.0) + (if (cueDir == null || cueNative != null) 0.03 else 0.0)
        val confidence = ((guideQ * 0.30 + cueQ * 0.29 + targetQ * 0.27 + refinedTargetHit.score * 0.14 + nativeBonus) * 100.0)
            .toInt().coerceIn(50, 98)

        val status = when {
            objectNative != null && (cueDir == null || cueNative != null) -> "ML 6.0: геометрия + штатная коррекция"
            else -> "ML 6.0: геометрия по центрам шаров"
        }
        val diagnostic = diagnosticBase(
            guides.size,
            balls.size,
            cueGuide,
            refinedTargetHit,
            objectNative,
            cueNative
        ) + "; normalSpeed=${fmt(normalSpeed)}; cueResidual=${fmt(cueResidual)}" +
            "; ghost=${fmt(ghost.x)},${fmt(ghost.y)}" +
            "; objDir=${fmt(objectDir.x)},${fmt(objectDir.y)}" +
            (cueDir?.let { "; cueDir=${fmt(it.x)},${fmt(it.y)}" } ?: "; cueDir=none")

        gray.release(); hsv.release(); whiteMask.release(); edges.release(); table.release(); work.release()
        return AnalysisResult(
            bitmap.width,
            bitmap.height,
            cueGlobal,
            debugBalls,
            ghostGlobal,
            segmentsGlobal,
            playRect,
            confidence,
            status,
            targetBall = targetGlobal,
            aimDirection = toGlobalDirection(aimDir),
            diagnostics = diagnostic
        )
    }

    private fun fixedTable(work: Mat): Rect? {
        if (work.cols() < 300 || work.rows() < 200) return null
        val x = (work.cols() * 0.1787).toInt().coerceIn(0, work.cols() - 2)
        val y = (work.rows() * 0.2080).toInt().coerceIn(0, work.rows() - 2)
        val right = (work.cols() * 0.8215).toInt().coerceIn(x + 2, work.cols())
        val bottom = (work.rows() * 0.9410).toInt().coerceIn(y + 2, work.rows())
        return Rect(x, y, right - x, bottom - y)
    }

    private fun localPlayRect(table: Mat) = PlayRect(
        0.0,
        0.0,
        (table.cols() - 1).toDouble(),
        (table.rows() - 1).toDouble()
    )

    private fun buildWhiteMask(table: Mat): Mat {
        val rgb = Mat()
        val hsv = Mat()
        val mask = Mat()
        Imgproc.cvtColor(table, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
        rgb.release()
        // Slightly stricter saturation than 4.2.1 so yellow/cyan overlay feedback is
        // excluded from the detector while the game's white guide remains.
        Core.inRange(hsv, Scalar(0.0, 0.0, 152.0), Scalar(179.0, 74.0, 255.0), mask)
        hsv.release()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel)
        kernel.release()
        val border = max(7, (table.cols() * 0.010).toInt())
        Imgproc.rectangle(
            mask,
            Point(0.0, 0.0),
            Point((mask.cols() - 1).toDouble(), (mask.rows() - 1).toDouble()),
            Scalar(0.0),
            border
        )
        return mask
    }

    private fun detectGuides(mask: Mat, width: Double): List<Guide> {
        val lines = Mat()
        val minLen = max(58.0, width * 0.075)
        Imgproc.HoughLinesP(
            mask,
            lines,
            1.0,
            PI / 720.0,
            25,
            minLen,
            max(15.0, width * 0.020)
        )
        val out = mutableListOf<Guide>()
        for (row in 0 until lines.rows()) {
            val l = lines.get(row, 0) ?: continue
            if (l.size < 4) continue
            val a = Vec2(l[0], l[1])
            val b = Vec2(l[2], l[3])
            val len = (b - a).length()
            if (len < minLen) continue
            val midpoint = (a + b) * 0.5
            // Long table-edge fragments are already mostly removed by the border.
            // A soft centrality term keeps an interior native guide ahead of residual
            // rail highlights with a similar Hough length.
            val cx = mask.cols() * 0.5
            val cy = mask.rows() * 0.5
            val central = exp(-sqrt((midpoint.x - cx) * (midpoint.x - cx) + (midpoint.y - cy) * (midpoint.y - cy)) / max(1.0, width * 0.75))
            val q = ((len / max(1.0, width * 0.42)).coerceIn(0.35, 1.0) * 0.78 + central * 0.22)
                .coerceIn(0.0, 1.0)
            out += Guide(a, b, len, q)
        }
        lines.release()
        return out.sortedByDescending { it.length * (0.75 + it.quality * 0.25) }.take(20)
    }

    private fun detectBalls(gray: Mat, hsv: Mat, edges: Mat, width: Double): List<CircleCandidate> {
        val circles = Mat()
        val minR = max(7, (width * 0.0080).toInt())
        val maxR = max(minR + 3, (width * 0.0195).toInt())
        Imgproc.HoughCircles(
            gray,
            circles,
            Imgproc.HOUGH_GRADIENT,
            1.15,
            max(18.0, width * 0.021),
            108.0,
            16.0,
            minR,
            maxR
        )

        val out = mutableListOf<CircleCandidate>()
        val count = if (circles.rows() > 0) circles.cols() else 0
        val margin = max(18.0, width * 0.020)
        for (i in 0 until count) {
            val c = circles.get(0, i) ?: continue
            if (c.size < 3) continue
            val center = Vec2(c[0], c[1])
            val radius = c[2]
            if (center.x < margin || center.x > gray.cols() - margin ||
                center.y < margin || center.y > gray.rows() - margin) continue
            val edge = edgeCoverage(edges, center, radius)
            val white = interiorWhiteFraction(hsv, center, radius)
            val texture = interiorTexture(gray, center, radius)
            // The ghost guide is a white ring with felt in the middle: strong edge but
            // weak interior texture/whiteness. Reject that common false circle.
            if (edge < 0.18) continue
            if (texture < 0.14 && white < 0.34) continue
            out += CircleCandidate(center, radius, white, edge, texture)
        }
        circles.release()
        return out.sortedByDescending { it.quality }.take(28)
    }

    private fun edgeCoverage(edges: Mat, center: Vec2, radius: Double): Double {
        var hits = 0
        val samples = 40
        for (i in 0 until samples) {
            val a = 2.0 * PI * i / samples
            var hit = false
            for (dr in -2..2) {
                val rr = radius + dr
                val x = (center.x + cos(a) * rr).toInt()
                val y = (center.y + sin(a) * rr).toInt()
                if (x !in 0 until edges.cols() || y !in 0 until edges.rows()) continue
                val v = edges.get(y, x)
                if (v != null && v.isNotEmpty() && v[0] > 0.0) { hit = true; break }
            }
            if (hit) hits++
        }
        return hits.toDouble() / samples
    }

    private fun interiorWhiteFraction(hsv: Mat, center: Vec2, radius: Double): Double {
        val rr = max(2.0, radius * 0.60)
        val step = max(1, (radius / 5.0).toInt())
        var white = 0
        var total = 0
        var y = (center.y - rr).toInt()
        while (y <= (center.y + rr).toInt()) {
            var x = (center.x - rr).toInt()
            while (x <= (center.x + rr).toInt()) {
                val dx = x - center.x
                val dy = y - center.y
                if (dx * dx + dy * dy <= rr * rr && x in 0 until hsv.cols() && y in 0 until hsv.rows()) {
                    val p = hsv.get(y, x)
                    if (p != null && p.size >= 3) {
                        total++
                        if (p[1] <= 72.0 && p[2] >= 145.0) white++
                    }
                }
                x += step
            }
            y += step
        }
        return white.toDouble() / max(1, total).toDouble()
    }

    private fun interiorTexture(gray: Mat, center: Vec2, radius: Double): Double {
        val rr = max(2.0, radius * 0.62)
        val step = max(1, (radius / 5.0).toInt())
        var n = 0
        var sum = 0.0
        var sum2 = 0.0
        var y = (center.y - rr).toInt()
        while (y <= (center.y + rr).toInt()) {
            var x = (center.x - rr).toInt()
            while (x <= (center.x + rr).toInt()) {
                val dx = x - center.x
                val dy = y - center.y
                if (dx * dx + dy * dy <= rr * rr && x in 0 until gray.cols() && y in 0 until gray.rows()) {
                    val p = gray.get(y, x)
                    if (p != null && p.isNotEmpty()) {
                        n++
                        sum += p[0]
                        sum2 += p[0] * p[0]
                    }
                }
                x += step
            }
            y += step
        }
        if (n < 4) return 0.0
        val mean = sum / n
        val std = sqrt(max(0.0, sum2 / n - mean * mean))
        return (std / 28.0).coerceIn(0.0, 1.0)
    }

    private fun mergeBallPool(
        fresh: List<CircleCandidate>,
        cached: List<CircleCandidate>,
        width: Double
    ): List<CircleCandidate> {
        if (cached.isEmpty()) return fresh
        if (fresh.isEmpty()) return cached
        val out = fresh.toMutableList()
        val maxNear = max(8.0, width * 0.012)
        for (old in cached) {
            if (out.none { (it.center - old.center).length() <= max(maxNear, (it.radius + old.radius) * 0.55) }) {
                out += old
            }
        }
        return out.sortedByDescending { it.quality }.take(30)
    }

    private fun chooseCueAndGuide(
        guides: List<Guide>,
        balls: List<CircleCandidate>,
        width: Double
    ): CueGuide? {
        if (balls.isEmpty()) return null
        var best: CueGuide? = null
        var bestScore = Double.NEGATIVE_INFINITY
        for (g in guides.take(14)) {
            val lineDir = (g.b - g.a).normalized()
            if (lineDir.length() < 0.5) continue
            for (c in balls) {
                val lineDist = abs((c.center - g.a).cross(lineDir))
                if (lineDist > max(c.radius * 0.95, width * 0.0085)) continue
                val proj = (c.center - g.a).dot(lineDir)
                if (proj < -width * 0.055 || proj > g.length + width * 0.055) continue
                val da = (c.center - g.a).length()
                val db = (c.center - g.b).length()
                val far = if (da >= db) g.a else g.b
                val near = if (da < db) g.a else g.b
                var dir = (far - c.center).normalized()
                if (dir.length() < 0.5) continue
                // If the nearest endpoint is not actually the cue-side endpoint, flip
                // according to projection toward the longer visible portion.
                if ((far - near).dot(dir) < 0.0) dir = dir * -1.0
                val endDist = min(da, db)
                val whiteQ = c.whiteFraction
                val cuePrior = lastCueCenter?.let {
                    exp(-(c.center - it).length() / max(1.0, width * 0.030))
                } ?: 0.0
                val score =
                    g.quality * 2.2 +
                    whiteQ * 3.8 +
                    c.edgeCoverage * 0.9 +
                    exp(-lineDist / max(1.0, c.radius * 0.75)) * 1.5 +
                    exp(-endDist / max(1.0, width * 0.055)) * 1.0 +
                    cuePrior * 1.1
                if (score > bestScore) {
                    bestScore = score
                    best = CueGuide(c, g, dir, score)
                }
            }
        }
        val chosen = best ?: return null
        // Very low achromatic score usually means a striped/colored target was used
        // as the cue endpoint. Allow some shading/red-dot loss but reject clear color.
        if (chosen.cue.whiteFraction < 0.24 && chosen.score < 5.0) return null
        return chosen
    }

    private fun chooseTarget(
        cue: CircleCandidate,
        direction: Vec2,
        balls: List<CircleCandidate>,
        width: Double
    ): TargetHit? {
        var best: TargetHit? = null
        var bestT = Double.POSITIVE_INFINITY
        for (b in balls) {
            if ((b.center - cue.center).length() < max(cue.radius, b.radius) * 1.15) continue
            val hit = rayCircleHit(cue, direction, b, width) ?: continue
            if (hit.t < bestT) {
                bestT = hit.t
                best = hit
            }
        }
        return best
    }

    private fun rayCircleHit(
        cue: CircleCandidate,
        direction: Vec2,
        target: CircleCandidate,
        width: Double
    ): TargetHit? {
        val d = direction.normalized()
        if (d.length() < 0.5) return null
        val sumR = (cue.radius + target.radius).coerceAtLeast(width * 0.015)
        val oc = cue.center - target.center
        val b = 2.0 * d.dot(oc)
        val c = oc.dot(oc) - sumR * sumR
        val disc = b * b - 4.0 * c
        if (disc < 0.0) return null
        val root = sqrt(max(0.0, disc))
        val t1 = (-b - root) * 0.5
        val t2 = (-b + root) * 0.5
        val t = when {
            t1 > cue.radius * 0.35 -> t1
            t2 > cue.radius * 0.35 -> t2
            else -> return null
        }
        if (t > width * 1.35) return null
        val ghost = cue.center + d * t
        val normal = (target.center - ghost).normalized()
        if (normal.length() < 0.5) return null
        val alignment = d.dot(normal).coerceIn(0.0, 1.0)
        // A near-tangent contact is valid, but lower target Hough quality should not
        // beat a strong closer ball. Distance remains the primary selector.
        val q = (target.quality * 0.66 + alignment * 0.20 + exp(-t / max(1.0, width * 0.75)) * 0.14)
            .coerceIn(0.0, 1.0)
        return TargetHit(target, t, ghost, normal, q)
    }

    private fun refineGhostCenter(
        mask: Mat,
        predicted: Vec2,
        cueRadius: Double,
        width: Double
    ): Vec2 {
        val search = max(3, (width * 0.0045).toInt()).coerceAtMost(7)
        val ringR = cueRadius.coerceAtLeast(width * 0.008)
        var best = predicted
        var bestScore = ringScore(mask, predicted, ringR)
        for (dy in -search..search) {
            for (dx in -search..search) {
                if (dx == 0 && dy == 0) continue
                val p = Vec2(predicted.x + dx, predicted.y + dy)
                val score = ringScore(mask, p, ringR) - sqrt((dx * dx + dy * dy).toDouble()) * 0.010
                if (score > bestScore) {
                    bestScore = score
                    best = p
                }
            }
        }
        // Require a real ring response; otherwise analytic geometry stays authoritative.
        return if (bestScore >= 0.25 && (best - predicted).length() <= max(6.0, width * 0.007)) best else predicted
    }

    private fun ringScore(mask: Mat, center: Vec2, radius: Double): Double {
        var ringHits = 0
        var ringSamples = 0
        for (i in 0 until 36) {
            val a = 2.0 * PI * i / 36.0
            var hit = false
            for (dr in -2..2) {
                val rr = radius + dr
                val x = (center.x + cos(a) * rr).toInt()
                val y = (center.y + sin(a) * rr).toInt()
                if (x !in 0 until mask.cols() || y !in 0 until mask.rows()) continue
                ringSamples++
                val v = mask.get(y, x)
                if (v != null && v.isNotEmpty() && v[0] > 0.0) hit = true
            }
            if (hit) ringHits++
        }
        val ringQ = ringHits.toDouble() / 36.0
        val centerWhite = whiteFractionMask(mask, center, max(2, (radius * 0.38).toInt()))
        return (ringQ * 0.88 - centerWhite * 0.18).coerceIn(0.0, 1.0)
    }

    private fun refineExpectedBranch(
        mask: Mat,
        ghost: Vec2,
        expected: Vec2,
        incoming: Vec2,
        width: Double,
        searchDegrees: Double
    ): RayEvidence? {
        val e = expected.normalized()
        if (e.length() < 0.5) return null
        var best: RayEvidence? = null
        var delta = -searchDegrees
        while (delta <= searchDegrees + 1e-6) {
            val d = rotate(e, Math.toRadians(delta)).normalized()
            // Outgoing branches should not point strongly backwards into the cue.
            if (d.dot(incoming) >= -0.18) {
                val evidence = rayWhiteEvidence(mask, ghost, d, width)
                val angularPrior = exp(-abs(delta) / max(1.0, searchDegrees * 0.58))
                val combined = evidence.score * 0.82 + angularPrior * 0.18
                val candidate = evidence.copy(direction = d, score = combined)
                if (best == null || candidate.score > best!!.score) best = candidate
            }
            delta += 0.5
        }
        val chosen = best ?: return null
        return if (chosen.score >= 0.34 && chosen.lastWhite >= max(10.0, width * 0.012)) chosen else null
    }

    private fun rayWhiteEvidence(mask: Mat, origin: Vec2, dir: Vec2, width: Double): RayEvidence {
        val inner = max(7.0, width * 0.008)
        val outer = max(58.0, width * 0.095)
        val perp = Vec2(-dir.y, dir.x)
        val corridor = max(2.2, width * 0.0034)
        var r = inner
        var hits = 0
        var samples = 0
        var lastHit = 0.0
        var longestRun = 0
        var run = 0
        while (r <= outer) {
            val p = origin + dir * r
            val hit = whiteAcross(mask, p, perp, corridor)
            samples++
            if (hit) {
                hits++
                run++
                longestRun = max(longestRun, run)
                lastHit = r
            } else {
                run = 0
            }
            r += 1.25
        }
        val ratio = hits.toDouble() / max(1, samples).toDouble()
        val runQ = (longestRun / max(1.0, samples * 0.34)).coerceIn(0.0, 1.0)
        val spanQ = (lastHit / max(1.0, outer * 0.72)).coerceIn(0.0, 1.0)
        val score = (ratio * 0.48 + runQ * 0.32 + spanQ * 0.20).coerceIn(0.0, 1.0)
        return RayEvidence(dir, score, lastHit)
    }

    private fun branchRenderStart(
        mask: Mat,
        ghost: Vec2,
        dir: Vec2,
        native: RayEvidence?,
        cueRadius: Double,
        targetRadius: Double,
        width: Double,
        isObject: Boolean
    ): Vec2 {
        val measured = native?.lastWhite ?: findLastWhiteAlong(mask, ghost, dir, width)
        val fallback = if (isObject) {
            cueRadius + targetRadius + targetRadius * 0.78
        } else {
            cueRadius * 2.15
        }
        val distance = max(fallback, measured + max(1.5, width * 0.002))
            .coerceAtMost(width * 0.125)
        return ghost + dir.normalized() * distance
    }

    private fun findLastWhiteAlong(mask: Mat, origin: Vec2, dir: Vec2, width: Double): Double {
        val perp = Vec2(-dir.y, dir.x)
        val inner = max(7.0, width * 0.008)
        val outer = max(56.0, width * 0.095)
        val corridor = max(2.2, width * 0.0034)
        var r = inner
        var last = 0.0
        while (r <= outer) {
            if (whiteAcross(mask, origin + dir * r, perp, corridor)) last = r
            r += 1.25
        }
        return last
    }

    private fun whiteAcross(mask: Mat, p: Vec2, perp: Vec2, corridor: Double): Boolean {
        var hits = 0
        var samples = 0
        var o = -corridor
        while (o <= corridor) {
            val q = p + perp * o
            val x = q.x.toInt()
            val y = q.y.toInt()
            if (x in 0 until mask.cols() && y in 0 until mask.rows()) {
                samples++
                val v = mask.get(y, x)
                if (v != null && v.isNotEmpty() && v[0] > 0.0) hits++
            }
            o += 1.2
        }
        return hits >= max(2, samples / 3)
    }

    private fun whiteFractionMask(mask: Mat, center: Vec2, radius: Int): Double {
        var hits = 0
        var total = 0
        val r2 = radius * radius
        for (dy in -radius..radius step 2) {
            for (dx in -radius..radius step 2) {
                if (dx * dx + dy * dy > r2) continue
                val x = center.x.toInt() + dx
                val y = center.y.toInt() + dy
                if (x !in 0 until mask.cols() || y !in 0 until mask.rows()) continue
                total++
                val v = mask.get(y, x)
                if (v != null && v.isNotEmpty() && v[0] > 0.0) hits++
            }
        }
        return hits.toDouble() / max(1, total).toDouble()
    }

    private fun traceToCushions(
        drawStart: Vec2,
        initialDirection: Vec2,
        rawRect: PlayRect,
        ballRadius: Double,
        firstKind: SegmentKind,
        bounceCount: Int
    ): List<TrajectorySegment> {
        val rect = rawRect.inset(ballRadius)
        val out = mutableListOf<TrajectorySegment>()
        var start = clamp(drawStart, rect)
        var rayStart = start
        var dir = initialDirection.normalized()
        if (dir.length() < 0.5) return out

        for (leg in 0..bounceCount) {
            val hit = hitRect(rayStart, dir, rect) ?: break
            if ((hit.point - start).length() < 2.0) break
            out += TrajectorySegment(start, hit.point, if (leg == 0) firstKind else SegmentKind.BOUNCE)
            if (leg == bounceCount) break
            dir = Vec2(
                if (hit.vertical) -dir.x else dir.x,
                if (hit.horizontal) -dir.y else dir.y
            ).normalized()
            start = hit.point
            rayStart = hit.point + dir * 0.85
        }
        return out
    }

    private fun hitRect(p: Vec2, d: Vec2, rect: PlayRect): HitRect? {
        val eps = 1e-8
        var bestT = Double.POSITIVE_INFINITY
        fun consider(t: Double) {
            if (t > 1e-4 && t < bestT) bestT = t
        }
        if (d.x > eps) consider((rect.right - p.x) / d.x)
        else if (d.x < -eps) consider((rect.left - p.x) / d.x)
        if (d.y > eps) consider((rect.bottom - p.y) / d.y)
        else if (d.y < -eps) consider((rect.top - p.y) / d.y)
        if (!bestT.isFinite()) return null
        val hit = p + d * bestT
        if (hit.x < rect.left - 1.5 || hit.x > rect.right + 1.5 ||
            hit.y < rect.top - 1.5 || hit.y > rect.bottom + 1.5) return null
        val vertical = abs(hit.x - rect.left) <= 1.5 || abs(hit.x - rect.right) <= 1.5
        val horizontal = abs(hit.y - rect.top) <= 1.5 || abs(hit.y - rect.bottom) <= 1.5
        if (!vertical && !horizontal) return null
        return HitRect(clamp(hit, rect), vertical, horizontal)
    }

    private fun diagnosticBase(
        guideCount: Int,
        ballCount: Int,
        cueGuide: CueGuide,
        target: TargetHit?,
        objectNative: RayEvidence?,
        cueNative: RayEvidence?
    ): String {
        return buildString {
            append("guides=").append(guideCount)
            append("; balls=").append(ballCount)
            append("; cue=").append(fmt(cueGuide.cue.center.x)).append(',').append(fmt(cueGuide.cue.center.y))
            append(" r=").append(fmt(cueGuide.cue.radius))
            append(" white=").append(fmt(cueGuide.cue.whiteFraction))
            append(" edge=").append(fmt(cueGuide.cue.edgeCoverage))
            append("; aim=").append(fmt(cueGuide.direction.x)).append(',').append(fmt(cueGuide.direction.y))
            append("; guideLen=").append(fmt(cueGuide.guide.length))
            if (target != null) {
                append("; target=").append(fmt(target.target.center.x)).append(',').append(fmt(target.target.center.y))
                append(" r=").append(fmt(target.target.radius))
                append(" q=").append(fmt(target.target.quality))
                append(" t=").append(fmt(target.t))
            } else append("; target=none")
            append("; objNative=").append(objectNative?.let { fmt(it.score) } ?: "none")
            append("; cueNative=").append(cueNative?.let { fmt(it.score) } ?: "none")
        }
    }

    private fun toGlobal(p: Vec2, tableRect: Rect, scale: Double) = Vec2(
        (tableRect.x + p.x) / scale,
        (tableRect.y + p.y) / scale
    )

    private fun toGlobalDirection(d: Vec2) = d.normalized()

    private fun CircleCandidate.toBall(tableRect: Rect, scale: Double, cue: Boolean) = Ball(
        toGlobal(center, tableRect, scale),
        radius / scale,
        cue
    )

    private fun TrajectorySegment.toGlobal(tableRect: Rect, scale: Double) = TrajectorySegment(
        toGlobal(start, tableRect, scale),
        toGlobal(end, tableRect, scale),
        kind,
        dashed
    )

    private fun clamp(p: Vec2, rect: PlayRect) = Vec2(
        p.x.coerceIn(rect.left, rect.right),
        p.y.coerceIn(rect.top, rect.bottom)
    )

    private fun rotate(v: Vec2, a: Double): Vec2 {
        val c = cos(a)
        val s = sin(a)
        return Vec2(v.x * c - v.y * s, v.x * s + v.y * c)
    }

    private fun angleDeg(a: Vec2, b: Vec2): Double {
        val aa = atan2(a.y, a.x)
        val bb = atan2(b.y, b.x)
        var d = Math.toDegrees(aa - bb)
        while (d > 180.0) d -= 360.0
        while (d < -180.0) d += 360.0
        return abs(d)
    }

    private fun fmt(v: Double) = String.format(java.util.Locale.US, "%.3f", v)

    private fun clearCache() {
        cachedBalls = emptyList()
        cacheAtMs = 0L
        lastCueCenter = null
    }

    private fun emptyResult(bitmap: Bitmap, status: String) = AnalysisResult(
        bitmap.width,
        bitmap.height,
        null,
        emptyList(),
        null,
        emptyList(),
        null,
        0,
        status,
        diagnostics = "empty"
    )
}
