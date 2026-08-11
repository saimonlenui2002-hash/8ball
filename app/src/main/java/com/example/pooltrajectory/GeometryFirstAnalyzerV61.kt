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
 * 6.1 ghost-first validation geometry.
 *
 * Field diagnostics from 6.0 exposed two independent failure classes: the cue-stick
 * highlight could reverse the incoming Hough direction, and the game's own native
 * ghost ring could be mistaken for a target circle. 6.1 uses observables in their
 * physical order instead of trying to infer collision state from the first Hough
 * circle on a ray.
 *
 * Authority chain:
 *  - measure both directions of each cue-aligned axis; continuous native white guide
 *    support wins, with a short temporal 180-degree guard;
 *  - scan that ray for the native ghost ring (future cue centre at collision);
 *  - choose the real target roughly one cue+target radius from the ghost centre;
 *  - reject impossible radius ratios, backwards normals and overlapping ghost-circle
 *    candidates;
 *  - only if the native ghost cannot be resolved, use a conservative ENTRY-root ray
 *    collision fallback;
 *  - native post-contact pixels can make only a small correction around the already
 *    physics-derived object/cue branches.
 */
class GeometryFirstAnalyzerV61(private val context: Context) {
    private val prefs = context.getSharedPreferences(MainActivity.PREFS, Context.MODE_PRIVATE)

    private data class CircleCandidate(
        val center: Vec2,
        val radius: Double,
        val whiteFraction: Double,
        val edgeCoverage: Double,
        val texture: Double
    ) {
        val quality: Double
            get() = (edgeCoverage * 0.50 + texture * 0.32 + min(1.0, whiteFraction * 1.15) * 0.18)
                .coerceIn(0.0, 1.0)
    }

    private data class Guide(
        val a: Vec2,
        val b: Vec2,
        val length: Double,
        val quality: Double
    )

    private data class IncomingSupport(
        val score: Double,
        val lastWhite: Double,
        val longestRun: Double
    )

    private data class CueGuide(
        val cue: CircleCandidate,
        val guide: Guide,
        val direction: Vec2,
        val score: Double,
        val support: IncomingSupport,
        val reverseSupport: IncomingSupport,
        val temporalFlip: Boolean = false
    )

    private data class TargetHit(
        val target: CircleCandidate,
        val t: Double,
        val ghost: Vec2,
        val normal: Vec2,
        val score: Double,
        val radiusRatio: Double,
        val rawAlignment: Double
    )

    private data class GhostEvidence(
        val center: Vec2,
        val score: Double,
        val distance: Double,
        val ringQuality: Double,
        val centerWhite: Double
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
    private var lastAimDirection: Vec2? = null
    private var lastAimAtMs = 0L

    fun analyze(bitmap: Bitmap): AnalysisResult {
        frameSeq++
        val now = System.currentTimeMillis()
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)

        val scale = min(1.0, 1600.0 / max(1, src.cols()).toDouble())
        val work = Mat()
        if (scale < 0.999) Imgproc.resize(src, work, Size(), scale, scale, Imgproc.INTER_AREA)
        else src.copyTo(work)
        src.release()

        val tableRect = fixedTable(work)
        if (tableRect == null) {
            work.release()
            clearState()
            return emptyResult(bitmap, "ML 6.1: кадр слишком мал")
        }

        if (cacheFrameWidth != 0 && (cacheFrameWidth != bitmap.width || cacheFrameHeight != bitmap.height)) {
            clearState()
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
        Imgproc.Canny(gray, edges, 50.0, 148.0)

        val guides = detectGuides(whiteMask, table.cols().toDouble())
        if (guides.isEmpty()) {
            release(gray, hsv, whiteMask, edges, table, work)
            if (now - lastAimAtMs > 1400L) clearState()
            return AnalysisResult(
                bitmap.width, bitmap.height, null, emptyList(), null, emptyList(), playRect,
                16, "ML 6.1: ожидание штатной линии прицеливания",
                diagnostics = "guide=0; retained=${if (now - lastAimAtMs <= 1400L) 1 else 0}"
            )
        }

        val compatibleCache = cacheFrameWidth == bitmap.width &&
            cacheFrameHeight == bitmap.height &&
            cacheTableWidth == table.cols() && cacheTableHeight == table.rows() &&
            now - cacheAtMs <= 950L

        val runBallDetection = !compatibleCache || cachedBalls.size < 5 || frameSeq % 2L == 0L
        val freshBalls = if (runBallDetection) detectBalls(gray, hsv, edges, table.cols().toDouble()) else emptyList()
        val balls = mergeBallPool(freshBalls, if (compatibleCache) cachedBalls else emptyList(), table.cols().toDouble())
        if (freshBalls.size >= 3) {
            cachedBalls = mergeBallPool(freshBalls, if (compatibleCache) cachedBalls else emptyList(), table.cols().toDouble())
                .take(28)
            cacheFrameWidth = bitmap.width
            cacheFrameHeight = bitmap.height
            cacheTableWidth = table.cols()
            cacheTableHeight = table.rows()
            cacheAtMs = now
        }

        var cueGuide = chooseCueAndGuide(guides, balls, whiteMask, table.cols().toDouble(), now)
        if (cueGuide == null) {
            val dbg = if (prefs.getBoolean(MainActivity.KEY_DEBUG, false)) {
                guides.take(2).map { g ->
                    TrajectorySegment(toGlobal(g.a, tableRect, scale), toGlobal(g.b, tableRect, scale), SegmentKind.AIM)
                }
            } else emptyList()
            release(gray, hsv, whiteMask, edges, table, work)
            return AnalysisResult(
                bitmap.width, bitmap.height, null, emptyList(), null, dbg, playRect,
                28, "ML 6.1: биток/направление не подтверждены",
                diagnostics = "guides=${guides.size}; balls=${balls.size}; cue=0"
            )
        }

        // A true aim direction cannot jump by ~180 degrees between adjacent frames
        // while the same cue ball is stationary. This is a backup to the two-sided
        // white-ray measurement and specifically blocks cue-stick/Hough sign flips.
        val cueCenterNow = cueGuide.cue.center
        val closeCue = lastCueCenter?.let { (cueCenterNow - it).length() <= table.cols() * 0.035 } ?: false
        val recentAim = now - lastAimAtMs <= 900L
        val prevAim = lastAimDirection
        if (closeCue && recentAim && prevAim != null && cueGuide.direction.dot(prevAim) < -0.45) {
            cueGuide = cueGuide.copy(
                direction = cueGuide.direction * -1.0,
                support = cueGuide.reverseSupport,
                reverseSupport = cueGuide.support,
                temporalFlip = true
            )
        }

        lastCueCenter = cueGuide.cue.center
        lastAimDirection = cueGuide.direction
        lastAimAtMs = now

        // The game's own ghost ring is the most direct observable collision state:
        // it is the future cue-ball centre at contact. 6.0 only used a nearby ring
        // as a tiny correction AFTER choosing a target circle, which let the ring
        // itself become a false target. 6.1 reverses that authority chain.
        val nativeGhost = findNativeGhostAlongRay(
            whiteMask,
            cueGuide.cue.center,
            cueGuide.direction,
            cueGuide.cue.radius,
            table.cols().toDouble()
        )

        var aimDir = cueGuide.direction
        if (nativeGhost != null) {
            val toGhost = (nativeGhost.center - cueGuide.cue.center).normalized()
            if (toGhost.length() > 0.5 && angleDeg(toGhost, cueGuide.direction) <= 2.8) {
                aimDir = toGhost
            }
        }

        var contactSource = "nativeGhost"
        var targetHit = nativeGhost?.let {
            chooseTargetAroundGhost(
                cueGuide.cue,
                it,
                aimDir,
                balls,
                whiteMask,
                table.cols().toDouble()
            )
        }
        var ghost = nativeGhost?.center

        // Fallback is deliberately conservative. It uses only the ENTRY root and
        // same-size ball candidates, so the old "inside a ghost ring -> exit root"
        // failure cannot return.
        if (targetHit == null) {
            contactSource = "rayFallback"
            val rayHit = chooseTarget(cueGuide.cue, aimDir, balls, table.cols().toDouble())
            if (rayHit != null) {
                val ghostVisual = refineGhostCenter(
                    whiteMask, rayHit.ghost, cueGuide.cue.radius, table.cols().toDouble()
                )
                val visualAim = (ghostVisual - cueGuide.cue.center).normalized()
                if (visualAim.length() > 0.5 && angleDeg(visualAim, aimDir) <= 1.4) {
                    aimDir = visualAim
                }
                targetHit = rayCircleEntryHit(
                    cueGuide.cue, aimDir, rayHit.target, table.cols().toDouble()
                ) ?: rayHit
                ghost = targetHit.ghost
                if ((ghostVisual - ghost).length() <= max(4.0, table.cols() * 0.0050)) {
                    ghost = ghostVisual
                }
            }
        }

        if (targetHit == null || ghost == null) {
            val cueGlobal = cueGuide.cue.toBall(tableRect, scale, true)
            val debugSegments = if (prefs.getBoolean(MainActivity.KEY_DEBUG, false)) {
                val list = mutableListOf<TrajectorySegment>()
                list += TrajectorySegment(
                    cueGlobal.center,
                    toGlobal(cueGuide.cue.center + aimDir * (table.cols() * 0.16), tableRect, scale),
                    SegmentKind.AIM
                )
                nativeGhost?.let { g ->
                    list += TrajectorySegment(
                        cueGlobal.center,
                        toGlobal(g.center, tableRect, scale),
                        SegmentKind.AIM
                    )
                }
                list
            } else emptyList()
            val diag = diagnosticBase(guides.size, balls.size, cueGuide, null, null, null) +
                "; source=${if (nativeGhost != null) "nativeGhostNoTarget" else "noGhost"}" +
                (nativeGhost?.let {
                    "; ghostScore=${fmt(it.score)}; ghostRing=${fmt(it.ringQuality)}; ghostCenterWhite=${fmt(it.centerWhite)}; ghostDist=${fmt(it.distance)}"
                } ?: "")
            release(gray, hsv, whiteMask, edges, table, work)
            return AnalysisResult(
                bitmap.width, bitmap.height, cueGlobal, listOf(cueGlobal),
                nativeGhost?.let { toGlobal(it.center, tableRect, scale) },
                debugSegments, playRect,
                if (nativeGhost != null) 48 else 42,
                if (nativeGhost != null) "ML 6.1: ghost найден, реальный шар не подтверждён"
                else "ML 6.1: цель на корректном луче не подтверждена",
                aimDirection = aimDir,
                diagnostics = diag
            )
        }

        // Keep the native ghost as the collision junction when available. For the
        // fallback path, a nearby native ring may still make a small visual snap.
        var normal = (targetHit.target.center - ghost).normalized()
        if (normal.length() < 0.5) normal = targetHit.normal

        val finalTarget = targetHit
        val rawNormalSpeed = aimDir.dot(normal)
        if (rawNormalSpeed < -0.015) {
            val cueGlobal = cueGuide.cue.toBall(tableRect, scale, true)
            val targetGlobal = finalTarget.target.toBall(tableRect, scale, false)
            release(gray, hsv, whiteMask, edges, table, work)
            return AnalysisResult(
                bitmap.width, bitmap.height, cueGlobal, listOf(cueGlobal, targetGlobal), null, emptyList(), playRect,
                32, "ML 6.1: отброшена невозможная геометрия контакта",
                targetBall = targetGlobal,
                aimDirection = aimDir,
                diagnostics = diagnosticBase(guides.size, balls.size, cueGuide, finalTarget, null, null) +
                    "; reject=negativeNormal; rawNormal=${fmt(rawNormalSpeed)}"
            )
        }

        val normalSpeed = rawNormalSpeed.coerceIn(0.0, 1.0)
        val objectPhysics = normal
        val cueVector = aimDir - normal * normalSpeed
        val cueResidual = cueVector.length()
        val cuePhysics = if (cueResidual >= 0.055) cueVector.normalized() else null

        val objectNative = if (normalSpeed >= 0.055) {
            refineExpectedBranch(
                whiteMask, ghost, objectPhysics, aimDir,
                table.cols().toDouble(), searchDegrees = 8.0
            )
        } else null
        val objectDir = objectNative?.direction ?: objectPhysics

        val cueNative = cuePhysics?.let {
            refineExpectedBranch(
                whiteMask, ghost, it, aimDir,
                table.cols().toDouble(), searchDegrees = 10.0
            )
        }
        val cueDir = cueNative?.direction ?: cuePhysics

        val objectStart = if (normalSpeed >= 0.055) {
            branchRenderStart(
                whiteMask, ghost, objectDir, objectNative,
                cueGuide.cue.radius, finalTarget.target.radius,
                table.cols().toDouble(), true
            )
        } else null
        val cueStart = cueDir?.let {
            branchRenderStart(
                whiteMask, ghost, it, cueNative,
                cueGuide.cue.radius, finalTarget.target.radius,
                table.cols().toDouble(), false
            )
        }

        val bounces = prefs.getInt(MainActivity.KEY_BOUNCES, 0).coerceIn(0, 2)
        val segmentsLocal = mutableListOf<TrajectorySegment>()
        if (objectStart != null && normalSpeed >= 0.055) {
            segmentsLocal += traceToCushions(
                objectStart, objectDir, localPlayRect(table), finalTarget.target.radius,
                SegmentKind.OBJECT, bounces
            )
        }
        if (cueDir != null && cueStart != null && cueResidual >= 0.075) {
            segmentsLocal += traceToCushions(
                cueStart, cueDir, localPlayRect(table), cueGuide.cue.radius,
                SegmentKind.CUE_AFTER, bounces
            )
        }

        if (prefs.getBoolean(MainActivity.KEY_DEBUG, false)) {
            segmentsLocal += TrajectorySegment(cueGuide.guide.a, cueGuide.guide.b, SegmentKind.AIM)
            segmentsLocal += TrajectorySegment(cueGuide.cue.center, ghost, SegmentKind.AIM)
        }

        val cueGlobal = cueGuide.cue.toBall(tableRect, scale, true)
        val targetGlobal = finalTarget.target.toBall(tableRect, scale, false)
        val ghostGlobal = toGlobal(ghost, tableRect, scale)
        val segmentsGlobal = segmentsLocal.map { it.toGlobal(tableRect, scale) }

        val cueQ = (cueGuide.cue.whiteFraction * 0.64 + cueGuide.cue.edgeCoverage * 0.36).coerceIn(0.0, 1.0)
        val targetQ = finalTarget.target.quality
        val radiusQ = (1.0 - abs(finalTarget.radiusRatio - 1.0) / 0.28).coerceIn(0.0, 1.0)
        val nativeBonus = (if (objectNative != null) 0.035 else 0.0) + (if (cueDir == null || cueNative != null) 0.025 else 0.0)
        val ghostQ = nativeGhost?.score ?: 0.0
        val confidence = ((cueGuide.guide.quality * 0.14 + cueQ * 0.18 +
            cueGuide.support.score * 0.18 + targetQ * 0.15 + radiusQ * 0.10 +
            finalTarget.score * 0.12 + ghostQ * 0.13 + nativeBonus) * 100.0).toInt().coerceIn(50, 98)

        val status = when {
            normalSpeed < 0.055 -> "ML 6.1: почти касательный контакт — цель подавлена"
            objectNative != null && (cueDir == null || cueNative != null) -> "ML 6.1: ghost-геометрия + штатная коррекция"
            else -> "ML 6.1: ghost/центр геометрия"
        }
        val diagnostic = diagnosticBase(
            guides.size, balls.size, cueGuide, finalTarget, objectNative, cueNative
        ) + "; normalSpeed=${fmt(normalSpeed)}; cueResidual=${fmt(cueResidual)}" +
            "; source=$contactSource" +
            (nativeGhost?.let {
                "; ghostScore=${fmt(it.score)}; ghostRing=${fmt(it.ringQuality)}; ghostCenterWhite=${fmt(it.centerWhite)}; ghostDist=${fmt(it.distance)}"
            } ?: "; ghostScore=none") +
            "; ghost=${fmt(ghost.x)},${fmt(ghost.y)}" +
            "; objDir=${fmt(objectDir.x)},${fmt(objectDir.y)}" +
            (cueDir?.let { "; cueDir=${fmt(it.x)},${fmt(it.y)}" } ?: "; cueDir=none")

        release(gray, hsv, whiteMask, edges, table, work)
        return AnalysisResult(
            bitmap.width, bitmap.height, cueGlobal, listOf(cueGlobal, targetGlobal), ghostGlobal,
            segmentsGlobal, playRect, confidence, status,
            targetBall = targetGlobal,
            aimDirection = aimDir,
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
        0.0, 0.0, (table.cols() - 1).toDouble(), (table.rows() - 1).toDouble()
    )

    private fun buildWhiteMask(table: Mat): Mat {
        val rgb = Mat()
        val hsv = Mat()
        val mask = Mat()
        Imgproc.cvtColor(table, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
        rgb.release()
        Core.inRange(hsv, Scalar(0.0, 0.0, 155.0), Scalar(179.0, 70.0, 255.0), mask)
        hsv.release()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel)
        kernel.release()
        val border = max(7, (table.cols() * 0.010).toInt())
        Imgproc.rectangle(
            mask, Point(0.0, 0.0), Point((mask.cols() - 1).toDouble(), (mask.rows() - 1).toDouble()),
            Scalar(0.0), border
        )
        return mask
    }

    private fun detectGuides(mask: Mat, width: Double): List<Guide> {
        val lines = Mat()
        val minLen = max(54.0, width * 0.068)
        Imgproc.HoughLinesP(
            mask, lines, 1.0, PI / 720.0, 23, minLen, max(14.0, width * 0.018)
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
            val cx = mask.cols() * 0.5
            val cy = mask.rows() * 0.5
            val central = exp(-sqrt((midpoint.x - cx) * (midpoint.x - cx) + (midpoint.y - cy) * (midpoint.y - cy)) /
                max(1.0, width * 0.75))
            val q = ((len / max(1.0, width * 0.42)).coerceIn(0.30, 1.0) * 0.76 + central * 0.24)
                .coerceIn(0.0, 1.0)
            out += Guide(a, b, len, q)
        }
        lines.release()
        return out.sortedByDescending { it.length * (0.72 + it.quality * 0.28) }.take(22)
    }

    private fun detectBalls(gray: Mat, hsv: Mat, edges: Mat, width: Double): List<CircleCandidate> {
        val circles = Mat()
        val minR = max(8, (width * 0.0090).toInt())
        val maxR = max(minR + 3, (width * 0.0173).toInt())
        Imgproc.HoughCircles(
            gray, circles, Imgproc.HOUGH_GRADIENT, 1.15,
            max(18.0, width * 0.021), 108.0, 16.0, minR, maxR
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
            if (edge < 0.20) continue
            // A native ghost ring has felt in its interior. Require either real
            // interior structure or substantial achromatic fill.
            if (texture < 0.17 && white < 0.38) continue
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
                        if (p[1] <= 70.0 && p[2] >= 148.0) white++
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
        fresh: List<CircleCandidate>, cached: List<CircleCandidate>, width: Double
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
        mask: Mat,
        width: Double,
        now: Long
    ): CueGuide? {
        if (balls.isEmpty()) return null
        var best: CueGuide? = null
        var bestScore = Double.NEGATIVE_INFINITY
        for (g in guides.take(18)) {
            val axis = (g.b - g.a).normalized()
            if (axis.length() < 0.5) continue
            for (c in balls) {
                // The cue ball in this game remains strongly achromatic even with
                // the red spot and shading. This rejects striped targets as cue.
                if (c.whiteFraction < 0.40) continue
                val lineDist = abs((c.center - g.a).cross(axis))
                if (lineDist > max(c.radius * 0.90, width * 0.0080)) continue
                val proj = (c.center - g.a).dot(axis)
                if (proj < -width * 0.060 || proj > g.length + width * 0.060) continue

                val plus = incomingWhiteEvidence(mask, c.center, axis, c.radius, width)
                val minus = incomingWhiteEvidence(mask, c.center, axis * -1.0, c.radius, width)
                var dir = if (plus.score >= minus.score) axis else axis * -1.0
                var forward = if (plus.score >= minus.score) plus else minus
                var reverse = if (plus.score >= minus.score) minus else plus

                val closeCue = lastCueCenter?.let { (c.center - it).length() <= width * 0.035 } ?: false
                val recent = now - lastAimAtMs <= 900L
                val prev = lastAimDirection
                var temporalFlip = false
                if (closeCue && recent && prev != null && dir.dot(prev) < -0.45) {
                    // When both directions have some white support, continuity wins
                    // over a one-frame Hough/cue-stick inversion.
                    if (forward.score <= reverse.score + 0.34) {
                        dir = dir * -1.0
                        val tmp = forward; forward = reverse; reverse = tmp
                        temporalFlip = true
                    }
                }

                if (forward.score < 0.16 && forward.longestRun < width * 0.050) continue
                val lineEndDist = min((c.center - g.a).length(), (c.center - g.b).length())
                val continuity = if (closeCue && recent && prev != null) ((dir.dot(prev) + 1.0) * 0.5).coerceIn(0.0, 1.0) else 0.0
                val score =
                    g.quality * 1.55 +
                    c.whiteFraction * 3.25 +
                    c.edgeCoverage * 0.75 +
                    exp(-lineDist / max(1.0, c.radius * 0.72)) * 1.25 +
                    exp(-lineEndDist / max(1.0, width * 0.060)) * 0.65 +
                    forward.score * 4.20 +
                    continuity * 1.10
                if (score > bestScore) {
                    bestScore = score
                    best = CueGuide(c, g, dir, score, forward, reverse, temporalFlip)
                }
            }
        }
        return best
    }

    private fun incomingWhiteEvidence(
        mask: Mat, origin: Vec2, dir: Vec2, cueRadius: Double, width: Double
    ): IncomingSupport {
        val d = dir.normalized()
        if (d.length() < 0.5) return IncomingSupport(0.0, 0.0, 0.0)
        val perp = Vec2(-d.y, d.x)
        val start = max(cueRadius * 1.02, width * 0.009)
        val outer = max(start + 45.0, width * 0.55)
        val step = 1.35
        var r = start
        var samples = 0
        var hits = 0
        var run = 0
        var longestRun = 0
        var gap = 0
        var lastHit = 0.0
        while (r <= outer) {
            val hit = whiteNarrow(mask, origin + d * r, perp, max(1.8, width * 0.0025))
            samples++
            if (hit) {
                hits++
                run++
                gap = 0
                longestRun = max(longestRun, run)
                lastHit = r
            } else if (run > 0 && gap < 1) {
                gap++
                run++
            } else {
                run = 0
                gap = 0
            }
            r += step
        }
        val ratio = hits.toDouble() / max(1, samples)
        val runPx = longestRun * step
        val runQ = (runPx / max(38.0, width * 0.19)).coerceIn(0.0, 1.0)
        val spanQ = ((lastHit - start) / max(45.0, width * 0.38)).coerceIn(0.0, 1.0)
        val score = (runQ * 0.58 + spanQ * 0.25 + ratio * 0.17).coerceIn(0.0, 1.0)
        return IncomingSupport(score, lastHit, runPx)
    }

    private fun whiteNarrow(mask: Mat, p: Vec2, perp: Vec2, corridor: Double): Boolean {
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
            o += 1.0
        }
        return hits >= max(1, samples / 3)
    }

    private fun findNativeGhostAlongRay(
        mask: Mat,
        cueCenter: Vec2,
        direction: Vec2,
        cueRadius: Double,
        width: Double
    ): GhostEvidence? {
        val d = direction.normalized()
        if (d.length() < 0.5) return null

        val start = max(cueRadius * 1.65, width * 0.020)
        var maxDistance = width * 1.25
        if (d.x > 1e-6) maxDistance = min(maxDistance, (mask.cols() - 2.0 - cueCenter.x) / d.x)
        else if (d.x < -1e-6) maxDistance = min(maxDistance, (1.0 - cueCenter.x) / d.x)
        if (d.y > 1e-6) maxDistance = min(maxDistance, (mask.rows() - 2.0 - cueCenter.y) / d.y)
        else if (d.y < -1e-6) maxDistance = min(maxDistance, (1.0 - cueCenter.y) / d.y)
        if (!maxDistance.isFinite() || maxDistance <= start + cueRadius) return null

        val ringR = cueRadius.coerceAtLeast(width * 0.008)
        var bestCenter: Vec2? = null
        var bestScore = Double.NEGATIVE_INFINITY
        var r = start
        while (r <= maxDistance) {
            val p = cueCenter + d * r
            val m = ringMetrics(mask, p, ringR)
            val distancePrior = exp(-r / max(1.0, width * 1.35))
            val score = m.first * 0.94 + distancePrior * 0.06
            if (score > bestScore) {
                bestScore = score
                bestCenter = p
            }
            r += 2.0
        }

        var center = bestCenter ?: return null
        val coarseCenter = center
        var metrics = ringMetrics(mask, center, ringR)
        var refinedScore = metrics.first
        for (dy in -4..4) {
            for (dx in -4..4) {
                val p = Vec2(coarseCenter.x + dx, coarseCenter.y + dy)
                if (p.x < 2.0 || p.x >= mask.cols() - 2.0 || p.y < 2.0 || p.y >= mask.rows() - 2.0) continue
                val m = ringMetrics(mask, p, ringR)
                val penalty = sqrt((dx * dx + dy * dy).toDouble()) * 0.006
                val score = m.first - penalty
                if (score > refinedScore) {
                    refinedScore = score
                    center = p
                    metrics = m
                }
            }
        }

        val distance = (center - cueCenter).dot(d)
        val score = metrics.first
        val ringQ = metrics.second
        val centerWhite = metrics.third
        if (score < 0.40 || ringQ < 0.44 || centerWhite > 0.62) return null
        if (distance < start - 5.0) return null
        return GhostEvidence(center, score, distance, ringQ, centerWhite)
    }

    private fun chooseTargetAroundGhost(
        cue: CircleCandidate,
        ghost: GhostEvidence,
        incoming: Vec2,
        balls: List<CircleCandidate>,
        mask: Mat,
        width: Double
    ): TargetHit? {
        var best: TargetHit? = null
        val inDir = incoming.normalized()
        if (inDir.length() < 0.5) return null

        for (b in balls) {
            if ((b.center - cue.center).length() < max(cue.radius, b.radius) * 1.12) continue
            val radiusRatio = b.radius / max(1e-6, cue.radius)
            if (radiusRatio < 0.72 || radiusRatio > 1.28) continue
            if (b.quality < 0.34) continue

            val delta = b.center - ghost.center
            val distance = delta.length()
            if (distance < 1e-6) continue
            val expectedDistance = cue.radius + b.radius
            val contactRatio = distance / max(1e-6, expectedDistance)
            if (contactRatio < 0.72 || contactRatio > 1.30) continue

            val normal = delta / distance
            val rawAlignment = inDir.dot(normal)
            if (rawAlignment < -0.03) continue

            val native = rayWhiteEvidence(mask, ghost.center, normal, width)
            val distanceQ = (1.0 - abs(contactRatio - 1.0) / 0.30).coerceIn(0.0, 1.0)
            val radiusQ = (1.0 - abs(radiusRatio - 1.0) / 0.28).coerceIn(0.0, 1.0)
            val forwardQ = ((rawAlignment + 0.03) / 0.55).coerceIn(0.0, 1.0)
            val score = (
                b.quality * 0.31 +
                distanceQ * 0.29 +
                native.score * 0.24 +
                radiusQ * 0.10 +
                forwardQ * 0.06
            ).coerceIn(0.0, 1.0)

            val t = (ghost.center - cue.center).dot(inDir).coerceAtLeast(0.0)
            val hit = TargetHit(b, t, ghost.center, normal, score, radiusRatio, rawAlignment)
            if (best == null || hit.score > best!!.score) best = hit
        }
        return best
    }

    private fun chooseTarget(
        cue: CircleCandidate,
        direction: Vec2,
        balls: List<CircleCandidate>,
        width: Double
    ): TargetHit? {
        var best: TargetHit? = null
        for (b in balls) {
            val centreDistance = (b.center - cue.center).length()
            if (centreDistance < max(cue.radius, b.radius) * 1.12) continue
            val ratio = b.radius / max(1e-6, cue.radius)
            if (ratio < 0.72 || ratio > 1.28) continue
            if (b.quality < 0.42) continue
            val hit = rayCircleEntryHit(cue, direction, b, width) ?: continue
            val current = best
            if (current == null ||
                hit.t < current.t - cue.radius * 0.45 ||
                (abs(hit.t - current.t) <= cue.radius * 0.45 && hit.score > current.score)) {
                best = hit
            }
        }
        return best
    }

    private fun rayCircleEntryHit(
        cue: CircleCandidate,
        direction: Vec2,
        target: CircleCandidate,
        width: Double
    ): TargetHit? {
        val d = direction.normalized()
        if (d.length() < 0.5) return null
        val sumR = (cue.radius + target.radius).coerceAtLeast(width * 0.015)
        val delta = target.center - cue.center
        val centerDistance = delta.length()
        // If the cue centre is well inside the inflated target circle, the circle
        // is almost certainly the game's ghost ring or another false Hough result.
        if (centerDistance < sumR * 0.92) return null
        val ahead = delta.dot(d)
        if (ahead < -cue.radius * 0.10) return null

        val oc = cue.center - target.center
        val qb = 2.0 * d.dot(oc)
        val qc = oc.dot(oc) - sumR * sumR
        val disc = qb * qb - 4.0 * qc
        if (disc < 0.0) return null
        val root = sqrt(max(0.0, disc))
        val entry = (-qb - root) * 0.5
        // Entry may be a fraction negative because Hough centres/radii jitter by a
        // pixel. Never fall through to the exit root: that was the 6.0 bug.
        if (entry < -cue.radius * 0.10) return null
        val t = max(0.0, entry)
        if (t > width * 1.35) return null

        val ghost = cue.center + d * t
        val normal = (target.center - ghost).normalized()
        if (normal.length() < 0.5) return null
        val rawAlignment = d.dot(normal)
        if (rawAlignment < -0.015) return null
        val alignment = rawAlignment.coerceIn(0.0, 1.0)
        val radiusRatio = target.radius / max(1e-6, cue.radius)
        val radiusQ = (1.0 - abs(radiusRatio - 1.0) / 0.28).coerceIn(0.0, 1.0)
        val q = (target.quality * 0.48 + alignment * 0.19 + radiusQ * 0.23 +
            exp(-t / max(1.0, width * 0.75)) * 0.10).coerceIn(0.0, 1.0)
        return TargetHit(target, t, ghost, normal, q, radiusRatio, rawAlignment)
    }

    private fun refineGhostCenter(mask: Mat, predicted: Vec2, cueRadius: Double, width: Double): Vec2 {
        val search = max(3, (width * 0.0035).toInt()).coerceAtMost(6)
        val ringR = cueRadius.coerceAtLeast(width * 0.008)
        var best = predicted
        var bestScore = ringScore(mask, predicted, ringR)
        for (dy in -search..search) {
            for (dx in -search..search) {
                if (dx == 0 && dy == 0) continue
                val p = Vec2(predicted.x + dx, predicted.y + dy)
                val score = ringScore(mask, p, ringR) - sqrt((dx * dx + dy * dy).toDouble()) * 0.012
                if (score > bestScore) { bestScore = score; best = p }
            }
        }
        return if (bestScore >= 0.28 && (best - predicted).length() <= max(5.0, width * 0.0055)) best else predicted
    }

    private fun ringMetrics(mask: Mat, center: Vec2, radius: Double): Triple<Double, Double, Double> {
        var ringHits = 0
        for (i in 0 until 36) {
            val a = 2.0 * PI * i / 36.0
            var hit = false
            for (dr in -2..2) {
                val rr = radius + dr
                val x = (center.x + cos(a) * rr).toInt()
                val y = (center.y + sin(a) * rr).toInt()
                if (x !in 0 until mask.cols() || y !in 0 until mask.rows()) continue
                val v = mask.get(y, x)
                if (v != null && v.isNotEmpty() && v[0] > 0.0) hit = true
            }
            if (hit) ringHits++
        }
        val ringQ = ringHits.toDouble() / 36.0
        val centerWhite = whiteFractionMask(mask, center, max(2, (radius * 0.38).toInt()))
        val score = (ringQ * 0.90 - centerWhite * 0.20).coerceIn(0.0, 1.0)
        return Triple(score, ringQ, centerWhite)
    }

    private fun ringScore(mask: Mat, center: Vec2, radius: Double): Double =
        ringMetrics(mask, center, radius).first

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
            if (d.dot(incoming) >= -0.12) {
                val evidence = rayWhiteEvidence(mask, ghost, d, width)
                val angularPrior = exp(-abs(delta) / max(1.0, searchDegrees * 0.52))
                val combined = evidence.score * 0.86 + angularPrior * 0.14
                val candidate = evidence.copy(direction = d, score = combined)
                if (best == null || candidate.score > best!!.score) best = candidate
            }
            delta += 0.5
        }
        val chosen = best ?: return null
        return if (chosen.score >= 0.44 && chosen.lastWhite >= max(12.0, width * 0.014)) chosen else null
    }

    private fun rayWhiteEvidence(mask: Mat, origin: Vec2, dir: Vec2, width: Double): RayEvidence {
        val inner = max(7.0, width * 0.008)
        val outer = max(54.0, width * 0.090)
        val perp = Vec2(-dir.y, dir.x)
        val corridor = max(2.0, width * 0.0030)
        var r = inner
        var hits = 0
        var samples = 0
        var lastHit = 0.0
        var longestRun = 0
        var run = 0
        while (r <= outer) {
            val hit = whiteAcross(mask, origin + dir * r, perp, corridor)
            samples++
            if (hit) {
                hits++; run++; longestRun = max(longestRun, run); lastHit = r
            } else run = 0
            r += 1.25
        }
        val ratio = hits.toDouble() / max(1, samples).toDouble()
        val runQ = (longestRun / max(1.0, samples * 0.32)).coerceIn(0.0, 1.0)
        val spanQ = (lastHit / max(1.0, outer * 0.72)).coerceIn(0.0, 1.0)
        return RayEvidence(dir, (ratio * 0.46 + runQ * 0.36 + spanQ * 0.18).coerceIn(0.0, 1.0), lastHit)
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
        val fallback = if (isObject) cueRadius + targetRadius + targetRadius * 1.00 else cueRadius * 2.20
        val distance = max(fallback, measured + max(2.0, width * 0.0022)).coerceAtMost(width * 0.120)
        return ghost + dir.normalized() * distance
    }

    private fun findLastWhiteAlong(mask: Mat, origin: Vec2, dir: Vec2, width: Double): Double {
        val perp = Vec2(-dir.y, dir.x)
        val inner = max(7.0, width * 0.008)
        val outer = max(52.0, width * 0.088)
        val corridor = max(2.0, width * 0.0030)
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
        fun consider(t: Double) { if (t > 1e-4 && t < bestT) bestT = t }
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
    ): String = buildString {
        append("guides=").append(guideCount)
        append("; balls=").append(ballCount)
        append("; cue=").append(fmt(cueGuide.cue.center.x)).append(',').append(fmt(cueGuide.cue.center.y))
        append(" r=").append(fmt(cueGuide.cue.radius))
        append(" white=").append(fmt(cueGuide.cue.whiteFraction))
        append(" edge=").append(fmt(cueGuide.cue.edgeCoverage))
        append("; aim=").append(fmt(cueGuide.direction.x)).append(',').append(fmt(cueGuide.direction.y))
        append("; guideLen=").append(fmt(cueGuide.guide.length))
        append("; inSupport=").append(fmt(cueGuide.support.score))
        append(" run=").append(fmt(cueGuide.support.longestRun))
        append("; reverseSupport=").append(fmt(cueGuide.reverseSupport.score))
        append("; temporalFlip=").append(if (cueGuide.temporalFlip) 1 else 0)
        if (target != null) {
            append("; target=").append(fmt(target.target.center.x)).append(',').append(fmt(target.target.center.y))
            append(" r=").append(fmt(target.target.radius))
            append(" q=").append(fmt(target.target.quality))
            append(" t=").append(fmt(target.t))
            append(" ratio=").append(fmt(target.radiusRatio))
            append(" align=").append(fmt(target.rawAlignment))
        } else append("; target=none")
        append("; objNative=").append(objectNative?.let { fmt(it.score) } ?: "none")
        append("; cueNative=").append(cueNative?.let { fmt(it.score) } ?: "none")
    }

    private fun toGlobal(p: Vec2, tableRect: Rect, scale: Double) = Vec2(
        (tableRect.x + p.x) / scale,
        (tableRect.y + p.y) / scale
    )

    private fun CircleCandidate.toBall(tableRect: Rect, scale: Double, cue: Boolean) = Ball(
        toGlobal(center, tableRect, scale), radius / scale, cue
    )

    private fun TrajectorySegment.toGlobal(tableRect: Rect, scale: Double) = TrajectorySegment(
        toGlobal(start, tableRect, scale), toGlobal(end, tableRect, scale), kind, dashed
    )

    private fun clamp(p: Vec2, rect: PlayRect) = Vec2(
        p.x.coerceIn(rect.left, rect.right), p.y.coerceIn(rect.top, rect.bottom)
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

    private fun release(vararg mats: Mat) {
        mats.forEach { runCatching { it.release() } }
    }

    private fun clearState() {
        cachedBalls = emptyList()
        cacheAtMs = 0L
        cacheFrameWidth = 0
        cacheFrameHeight = 0
        cacheTableWidth = 0
        cacheTableHeight = 0
        lastCueCenter = null
        lastAimDirection = null
        lastAimAtMs = 0L
    }

    private fun emptyResult(bitmap: Bitmap, status: String) = AnalysisResult(
        bitmap.width, bitmap.height, null, emptyList(), null, emptyList(), null, 0, status,
        diagnostics = "empty"
    )
}
