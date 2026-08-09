package com.example.pooltrajectory

import android.content.Context
import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.min

/**
 * Version 3 analyzer.
 *
 * Instead of detecting every ball, it trusts 8 Ball Pool's own aim guide:
 * 1) detect the felt rectangle,
 * 2) detect the long native white guide,
 * 3) find the fork at the collision marker,
 * 4) extend the short native post-collision branches to the cushions.
 */
class NativeGuideForkAnalyzer(private val context: Context) {
    private val prefs = context.getSharedPreferences(MainActivity.PREFS, Context.MODE_PRIVATE)

    private data class GuideSeg(val p1: Vec2, val p2: Vec2, val length: Double)
    private data class Branch(val direction: Vec2, val farPoint: Vec2, val length: Double)
    private data class Fork(val junction: Vec2, val incoming: Vec2, val branches: List<Branch>)

    fun analyze(bitmap: Bitmap): AnalysisResult {
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)

        val scale = min(1.0, 1280.0 / src.cols().toDouble())
        val work = Mat()
        if (scale < 0.999) {
            Imgproc.resize(src, work, Size(), scale, scale, Imgproc.INTER_AREA)
        } else {
            src.copyTo(work)
        }
        src.release()

        val tableRect = detectFeltRect(work)
        val table = work.submat(tableRect)
        val guideSegments = detectGuideSegments(table)
        val fork = findFork(guideSegments, table.cols().toDouble())

        val playRect = PlayRect(
            tableRect.x / scale,
            tableRect.y / scale,
            (tableRect.x + tableRect.width) / scale,
            (tableRect.y + tableRect.height) / scale
        )

        val out = mutableListOf<TrajectorySegment>()
        val debug = prefs.getBoolean(MainActivity.KEY_DEBUG, false)

        if (fork != null) {
            val ordered = fork.branches.sortedBy { branchAngle(fork.incoming, it.direction) }
            ordered.take(2).forEachIndexed { index, branch ->
                val kind = if (index == 0) SegmentKind.OBJECT else SegmentKind.CUE_AFTER
                val bounces = if (kind == SegmentKind.OBJECT) {
                    prefs.getInt(MainActivity.KEY_BOUNCES, 0).coerceIn(0, 2)
                } else {
                    0
                }
                out += extendBranch(
                    branch.farPoint,
                    branch.direction,
                    tableRect,
                    scale,
                    kind,
                    bounces
                )
            }
        } else if (debug) {
            // In diagnostics show what the white-line detector actually sees.
            guideSegments.take(12).forEach { s ->
                out += TrajectorySegment(
                    toGlobal(s.p1, tableRect, scale),
                    toGlobal(s.p2, tableRect, scale),
                    SegmentKind.AIM
                )
            }
        }

        table.release()
        work.release()

        val status = when {
            fork == null && guideSegments.isEmpty() -> "штатная белая линия не найдена"
            fork == null -> "развилка штатной линии не найдена"
            fork.branches.size == 1 -> "найдена 1 ветка после контакта"
            else -> "штатная развилка найдена"
        }
        val confidence = when {
            fork == null && guideSegments.isEmpty() -> 20
            fork == null -> 45
            fork.branches.size == 1 -> 75
            else -> 98
        }

        return AnalysisResult(
            bitmap.width,
            bitmap.height,
            null,
            emptyList(),
            null,
            out,
            playRect,
            confidence,
            status
        )
    }

    /**
     * Detects the felt using the dominant saturated hue in the center of the game.
     * This adapts to blue, cyan and green table skins without a huge HSV range that
     * accidentally absorbs UI elements.
     */
    private fun detectFeltRect(work: Mat): Rect {
        val rgb = Mat()
        val hsv = Mat()
        Imgproc.cvtColor(work, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
        rgb.release()

        val x0 = (work.cols() * 0.18).toInt().coerceIn(0, work.cols() - 1)
        val x1 = (work.cols() * 0.82).toInt().coerceIn(x0 + 1, work.cols())
        val y0 = (work.rows() * 0.22).toInt().coerceIn(0, work.rows() - 1)
        val y1 = (work.rows() * 0.88).toInt().coerceIn(y0 + 1, work.rows())

        val hist = IntArray(180)
        var yy = y0
        while (yy < y1) {
            var xx = x0
            while (xx < x1) {
                val p = hsv.get(yy, xx)
                if (p != null && p.size >= 3 && p[1] >= 60.0 && p[2] >= 65.0) {
                    hist[p[0].toInt().coerceIn(0, 179)]++
                }
                xx += 4
            }
            yy += 4
        }

        val dominantHue = hist.indices.maxByOrNull { hist[it] } ?: 95
        val mask = Mat()
        val low = dominantHue - 13
        val high = dominantHue + 13
        if (low >= 0 && high <= 179) {
            Core.inRange(hsv, Scalar(low.toDouble(), 45.0, 45.0), Scalar(high.toDouble(), 255.0, 255.0), mask)
        } else {
            val a = Mat()
            val b = Mat()
            if (low < 0) {
                Core.inRange(hsv, Scalar(0.0, 45.0, 45.0), Scalar(high.toDouble(), 255.0, 255.0), a)
                Core.inRange(hsv, Scalar((180 + low).toDouble(), 45.0, 45.0), Scalar(179.0, 255.0, 255.0), b)
            } else {
                Core.inRange(hsv, Scalar(low.toDouble(), 45.0, 45.0), Scalar(179.0, 255.0, 255.0), a)
                Core.inRange(hsv, Scalar(0.0, 45.0, 45.0), Scalar((high - 180).toDouble(), 255.0, 255.0), b)
            }
            Core.bitwise_or(a, b, mask)
            a.release()
            b.release()
        }

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(9.0, 9.0))
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel)
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel)
        kernel.release()

        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        val contourMask = mask.clone()
        Imgproc.findContours(contourMask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        val minArea = work.cols().toDouble() * work.rows().toDouble() * 0.18
        var bestRect: Rect? = null
        var bestArea = 0.0
        contours.forEach { c ->
            val area = Imgproc.contourArea(c)
            if (area >= minArea) {
                val r = Imgproc.boundingRect(c)
                val aspect = r.width.toDouble() / max(1, r.height).toDouble()
                if (aspect in 1.55..2.45 && area > bestArea) {
                    bestArea = area
                    bestRect = r
                }
            }
            c.release()
        }

        hierarchy.release()
        contourMask.release()
        mask.release()
        hsv.release()

        val fallback = Rect(
            (work.cols() * 0.18).toInt(),
            (work.rows() * 0.21).toInt(),
            (work.cols() * 0.64).toInt(),
            (work.rows() * 0.68).toInt()
        )
        val rect = bestRect ?: fallback
        rect.x = rect.x.coerceIn(0, work.cols() - 2)
        rect.y = rect.y.coerceIn(0, work.rows() - 2)
        rect.width = rect.width.coerceIn(2, work.cols() - rect.x)
        rect.height = rect.height.coerceIn(2, work.rows() - rect.y)
        return rect
    }

    private fun detectGuideSegments(tableRgba: Mat): List<GuideSeg> {
        val rgb = Mat()
        val hsv = Mat()
        val mask = Mat()
        Imgproc.cvtColor(tableRgba, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
        rgb.release()

        // White core of the native 8BP guideline.
        Core.inRange(hsv, Scalar(0.0, 0.0, 155.0), Scalar(179.0, 78.0, 255.0), mask)
        hsv.release()

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel)
        kernel.release()

        val border = max(5, (tableRgba.cols() * 0.008).toInt())
        Imgproc.rectangle(
            mask,
            Point(0.0, 0.0),
            Point((mask.cols() - 1).toDouble(), (mask.rows() - 1).toDouble()),
            Scalar(0.0),
            border
        )

        val lines = Mat()
        val minLen = max(20.0, tableRgba.cols() * 0.022)
        Imgproc.HoughLinesP(mask, lines, 1.0, Math.PI / 360.0, 18, minLen, 12.0)

        val out = mutableListOf<GuideSeg>()
        for (row in 0 until lines.rows()) {
            val l = lines.get(row, 0) ?: continue
            if (l.size < 4) continue
            val p1 = Vec2(l[0], l[1])
            val p2 = Vec2(l[2], l[3])
            val len = (p2 - p1).length()
            if (len < minLen) continue

            val mx = (p1.x + p2.x) * 0.5
            val my = (p1.y + p2.y) * 0.5
            val padX = tableRgba.cols() * 0.015
            val padY = tableRgba.rows() * 0.02
            if (mx < padX || mx > tableRgba.cols() - padX || my < padY || my > tableRgba.rows() - padY) continue

            out += GuideSeg(p1, p2, len)
        }

        lines.release()
        mask.release()
        return out.sortedByDescending { it.length }.take(80)
    }

    private fun findFork(segments: List<GuideSeg>, tableWidth: Double): Fork? {
        if (segments.isEmpty()) return null

        val longMin = max(70.0, tableWidth * 0.11)
        val branchMin = max(18.0, tableWidth * 0.018)
        val connectRadius = max(18.0, tableWidth * 0.022)

        var best: Fork? = null
        var bestScore = Double.NEGATIVE_INFINITY

        for (longSeg in segments.filter { it.length >= longMin }) {
            val endpoints = listOf(longSeg.p1 to longSeg.p2, longSeg.p2 to longSeg.p1)
            for ((junction, otherEnd) in endpoints) {
                val incoming = (junction - otherEnd).normalized()
                if (incoming.length() < 0.5) continue

                val raw = mutableListOf<Branch>()
                for (seg in segments) {
                    if (seg === longSeg || seg.length < branchMin) continue

                    val acute = acuteAngle(longSeg.p2 - longSeg.p1, seg.p2 - seg.p1)
                    if (acute < 10.0) continue

                    val d1 = (seg.p1 - junction).length()
                    val d2 = (seg.p2 - junction).length()
                    val nearest = min(d1, d2)
                    if (nearest > connectRadius) continue

                    val far = if (d1 >= d2) seg.p1 else seg.p2
                    val direction = (far - junction).normalized()
                    if (direction.length() < 0.5) continue

                    // Post-collision branches should not point back along the incoming ray.
                    if (direction.dot(incoming) < -0.20) continue
                    raw += Branch(direction, far, seg.length)
                }

                val branches = deduplicateBranches(raw)
                if (branches.isEmpty()) continue

                val score = longSeg.length +
                    branches.take(2).sumOf { it.length * 1.4 } +
                    if (branches.size >= 2) tableWidth * 0.22 else 0.0

                if (score > bestScore) {
                    bestScore = score
                    best = Fork(junction, incoming, branches.take(2))
                }
            }
        }
        return best
    }

    private fun deduplicateBranches(raw: List<Branch>): List<Branch> {
        val kept = mutableListOf<Branch>()
        for (candidate in raw.sortedByDescending { it.length }) {
            val duplicate = kept.any { branchAngle(it.direction, candidate.direction) < 9.0 }
            if (!duplicate) kept += candidate
        }
        return kept
    }

    private fun extendBranch(
        startLocal: Vec2,
        direction0: Vec2,
        tableRect: Rect,
        scale: Double,
        firstKind: SegmentKind,
        bounces: Int
    ): List<TrajectorySegment> {
        val result = mutableListOf<TrajectorySegment>()
        val estimatedRadius = tableRect.width * 0.012
        val bounds = PlayRect(
            estimatedRadius,
            estimatedRadius,
            tableRect.width - estimatedRadius,
            tableRect.height - estimatedRadius
        )

        var p = startLocal
        var d = direction0.normalized()
        val count = max(1, bounces + 1)
        repeat(count) { index ->
            val hit = rayToBoundary(p, d, bounds) ?: return@repeat
            val kind = if (index == 0) firstKind else SegmentKind.BOUNCE
            result += TrajectorySegment(
                toGlobal(p, tableRect, scale),
                toGlobal(hit.first, tableRect, scale),
                kind,
                firstKind == SegmentKind.CUE_AFTER || index > 0
            )
            if (index < bounces) {
                d = Vec2(if (hit.second) -d.x else d.x, if (hit.third) -d.y else d.y)
                p = hit.first + d * 0.8
            }
        }
        return result
    }

    private fun rayToBoundary(start: Vec2, d0: Vec2, rect: PlayRect): Triple<Vec2, Boolean, Boolean>? {
        val d = d0.normalized()
        val eps = 1e-7
        val ts = mutableListOf<Double>()
        if (abs(d.x) > eps) {
            ((rect.left - start.x) / d.x).takeIf { it > eps }?.let(ts::add)
            ((rect.right - start.x) / d.x).takeIf { it > eps }?.let(ts::add)
        }
        if (abs(d.y) > eps) {
            ((rect.top - start.y) / d.y).takeIf { it > eps }?.let(ts::add)
            ((rect.bottom - start.y) / d.y).takeIf { it > eps }?.let(ts::add)
        }
        if (ts.isEmpty()) return null

        val t = ts.min()
        val q = start + d * t
        val p = Vec2(
            q.x.coerceIn(rect.left, rect.right),
            q.y.coerceIn(rect.top, rect.bottom)
        )
        val hitX = abs(p.x - rect.left) < 2.0 || abs(p.x - rect.right) < 2.0
        val hitY = abs(p.y - rect.top) < 2.0 || abs(p.y - rect.bottom) < 2.0
        return Triple(p, hitX, hitY)
    }

    private fun toGlobal(local: Vec2, tableRect: Rect, scale: Double): Vec2 {
        return Vec2((local.x + tableRect.x) / scale, (local.y + tableRect.y) / scale)
    }

    private fun acuteAngle(a0: Vec2, b0: Vec2): Double {
        val a = a0.normalized()
        val b = b0.normalized()
        val dot = abs(a.dot(b)).coerceIn(0.0, 1.0)
        return Math.toDegrees(acos(dot))
    }

    private fun branchAngle(a0: Vec2, b0: Vec2): Double {
        val a = a0.normalized()
        val b = b0.normalized()
        val dot = a.dot(b).coerceIn(-1.0, 1.0)
        return Math.toDegrees(acos(dot))
    }
}
