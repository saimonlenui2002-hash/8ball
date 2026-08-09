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
import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * 4.1 ML geometry pass.
 *
 * The local ONNX model only says which pixels belong to the native guide/ring.
 * Geometry is solved deterministically:
 *  - fit a real circle center to the connected ML ring component;
 *  - re-fit the incoming native guide centerline from white pixels;
 *  - remove the incoming side and split outgoing ML pixels into connected components;
 *  - fit each short native branch with PCA using the component itself;
 *  - find the local target-ball circle and use center-to-center collision geometry
 *    to decide OBJECT (yellow) vs CUE_AFTER (blue), instead of angle ordering;
 *  - stabilize typed branches across a few steady frames.
 */
class NativeGuideMlAnalyzerV41(private val context: Context) {
    private val prefs = context.getSharedPreferences(MainActivity.PREFS, Context.MODE_PRIVATE)
    private val localDnn = LocalGuideDnn(context)
    private val history = ArrayDeque<TypedFrame>()

    private data class GuideSeg(val p1: Vec2, val p2: Vec2, val length: Double)
    private data class Fit(val direction: Vec2, val center: Vec2, val rms: Double, val linearity: Double)
    private data class RingFit(val center: Vec2, val radius: Double, val quality: Double)
    private data class RayBranch(
        val direction: Vec2,
        val start: Vec2,
        val end: Vec2,
        val score: Double,
        val length: Double,
        val quality: Double
    )
    private data class TargetBall(val center: Vec2, val radius: Double, val quality: Double)
    private data class LocalGeometry(
        val junction: Vec2,
        val incoming: Vec2,
        val incomingSeg: GuideSeg,
        val branches: List<RayBranch>,
        val ringQuality: Double,
        val geometryQuality: Double
    )
    private data class TypedFrame(
        val junction: Vec2,
        val objectBranch: RayBranch?,
        val cueBranch: RayBranch?,
        val targetCenter: Vec2?,
        val quality: Double,
        val rawBranches: List<RayBranch>,
        val incomingSeg: GuideSeg
    )
    private data class EndpointCandidate(val seed: Vec2, val other: Vec2, val seg: GuideSeg, val score: Double)

    fun analyze(bitmap: Bitmap): AnalysisResult {
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)
        val scale = min(1.0, 1280.0 / src.cols().toDouble())
        val work = Mat()
        if (scale < 0.999) Imgproc.resize(src, work, Size(), scale, scale, Imgproc.INTER_AREA) else src.copyTo(work)
        src.release()

        val tableRect = fixedTable(work)
        if (tableRect == null) {
            history.clear()
            work.release()
            return AnalysisResult(bitmap.width, bitmap.height, null, emptyList(), null, emptyList(), null, 0, "ML 4.1: кадр слишком мал")
        }
        val table = work.submat(tableRect)
        val playRect = PlayRect(
            tableRect.x / scale,
            tableRect.y / scale,
            (tableRect.x + tableRect.width) / scale,
            (tableRect.y + tableRect.height) / scale
        )

        // Broad cyan/green felt check. Unlike 3.x this does not require a single dominant hue,
        // but it reliably clears stale overlay on loading/connection/menu scenes.
        if (!looksLikePlayingTable(table)) {
            history.clear()
            table.release(); work.release()
            return AnalysisResult(bitmap.width, bitmap.height, null, emptyList(), null, emptyList(), playRect, 0, "ML 4.1: игровой стол не виден")
        }

        val whiteMask = buildWhiteMask(table)
        val longSegments = detectLongGuideSegments(whiteMask, table.cols())
        val geometry = findGeometry(table, whiteMask, longSegments, table.cols().toDouble())

        var typed: TypedFrame? = null
        if (geometry != null) {
            val target = detectTargetBall(table, geometry.junction, geometry.branches, table.cols().toDouble())
            typed = classify(geometry, target, table.cols().toDouble())
            if (typed != null) typed = stabilize(typed, table.cols().toDouble())
        } else {
            // No valid aim geometry in this frame: do not keep old trajectory on screen.
            history.clear()
        }

        val out = mutableListOf<TrajectorySegment>()
        val debug = prefs.getBoolean(MainActivity.KEY_DEBUG, false)
        if (typed != null) {
            typed.objectBranch?.let { b ->
                val bounces = prefs.getInt(MainActivity.KEY_BOUNCES, 0).coerceIn(0, 2)
                out += extendBranch(b.end, b.direction, tableRect, scale, SegmentKind.OBJECT, bounces)
            }
            typed.cueBranch?.let { b ->
                out += extendBranch(b.end, b.direction, tableRect, scale, SegmentKind.CUE_AFTER, 0)
            }

            if (debug) {
                out += TrajectorySegment(
                    toGlobal(typed.incomingSeg.p1, tableRect, scale),
                    toGlobal(typed.incomingSeg.p2, tableRect, scale),
                    SegmentKind.AIM
                )
                typed.rawBranches.forEach { b ->
                    out += TrajectorySegment(toGlobal(b.start, tableRect, scale), toGlobal(b.end, tableRect, scale), SegmentKind.AIM)
                }
                drawCross(out, typed.junction, tableRect, scale, max(6.0, tableRect.width * 0.0075))
                typed.targetCenter?.let { drawCross(out, it, tableRect, scale, max(5.0, tableRect.width * 0.006)) }
            }
        } else if (debug && geometry != null) {
            out += TrajectorySegment(
                toGlobal(geometry.incomingSeg.p1, tableRect, scale),
                toGlobal(geometry.incomingSeg.p2, tableRect, scale),
                SegmentKind.AIM
            )
            geometry.branches.forEach { b ->
                out += TrajectorySegment(toGlobal(b.start, tableRect, scale), toGlobal(b.end, tableRect, scale), SegmentKind.AIM)
            }
            drawCross(out, geometry.junction, tableRect, scale, max(6.0, tableRect.width * 0.0075))
        } else if (debug) {
            longSegments.take(4).forEach { s ->
                out += TrajectorySegment(toGlobal(s.p1, tableRect, scale), toGlobal(s.p2, tableRect, scale), SegmentKind.AIM)
            }
        }

        whiteMask.release(); table.release(); work.release()

        val status = when {
            longSegments.isEmpty() -> "ML 4.1: длинная линия не найдена"
            geometry == null && localDnn.lastError != null -> "ML 4.1: модель недоступна"
            geometry == null -> "ML 4.1: контакт/ветки не подтверждены"
            typed == null -> "ML 4.1: тип ветки не подтверждён"
            typed.objectBranch != null && typed.cueBranch != null -> "ML 4.1: объект + биток"
            typed.objectBranch != null -> "ML 4.1: прицельный шар"
            else -> "ML 4.1: белый шар"
        }
        val confidence = when {
            typed != null -> (typed.quality * 100.0).toInt().coerceIn(55, 99)
            geometry != null -> (geometry.geometryQuality * 70.0).toInt().coerceIn(35, 69)
            else -> 20
        }
        return AnalysisResult(bitmap.width, bitmap.height, null, emptyList(), null, out, playRect, confidence, status)
    }

    private fun fixedTable(work: Mat): Rect? {
        if (work.cols() < 300 || work.rows() < 200) return null
        val x = (work.cols() * 0.1787).toInt().coerceIn(0, work.cols() - 2)
        val y = (work.rows() * 0.2080).toInt().coerceIn(0, work.rows() - 2)
        val right = (work.cols() * 0.8215).toInt().coerceIn(x + 2, work.cols())
        val bottom = (work.rows() * 0.9410).toInt().coerceIn(y + 2, work.rows())
        return Rect(x, y, right - x, bottom - y)
    }

    private fun looksLikePlayingTable(table: Mat): Boolean {
        val rgb = Mat(); val hsv = Mat()
        Imgproc.cvtColor(table, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
        rgb.release()
        val x0 = (table.cols() * 0.16).toInt()
        val x1 = (table.cols() * 0.84).toInt()
        val y0 = (table.rows() * 0.20).toInt()
        val y1 = (table.rows() * 0.82).toInt()
        val step = max(3, table.cols() / 180)
        var felt = 0
        var total = 0
        var yy = y0
        while (yy < y1) {
            var xx = x0
            while (xx < x1) {
                val p = hsv.get(yy, xx)
                if (p != null && p.size >= 3) {
                    total++
                    val h = p[0]
                    if (p[1] >= 35.0 && p[2] >= 42.0 && h in 35.0..125.0) felt++
                }
                xx += step
            }
            yy += step
        }
        hsv.release()
        if (total < 100) return false
        return felt.toDouble() / total.toDouble() >= 0.43
    }

    private fun buildWhiteMask(table: Mat): Mat {
        val rgb = Mat(); val hsv = Mat(); val mask = Mat()
        Imgproc.cvtColor(table, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
        rgb.release()
        Core.inRange(hsv, Scalar(0.0, 0.0, 150.0), Scalar(179.0, 82.0, 255.0), mask)
        hsv.release()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel)
        kernel.release()
        val border = max(5, (table.cols() * 0.008).toInt())
        Imgproc.rectangle(mask, Point(0.0, 0.0), Point((mask.cols() - 1).toDouble(), (mask.rows() - 1).toDouble()), Scalar(0.0), border)
        return mask
    }

    private fun detectLongGuideSegments(mask: Mat, tableWidth: Int): List<GuideSeg> {
        val lines = Mat()
        val minLen = max(50.0, tableWidth * 0.065)
        Imgproc.HoughLinesP(mask, lines, 1.0, Math.PI / 360.0, 24, minLen, 18.0)
        val out = mutableListOf<GuideSeg>()
        for (row in 0 until lines.rows()) {
            val l = lines.get(row, 0) ?: continue
            if (l.size < 4) continue
            val p1 = Vec2(l[0], l[1]); val p2 = Vec2(l[2], l[3]); val len = (p2 - p1).length()
            if (len >= minLen) out += GuideSeg(p1, p2, len)
        }
        lines.release()
        return out.sortedByDescending { it.length }.take(16)
    }

    private fun findGeometry(table: Mat, mask: Mat, segments: List<GuideSeg>, tableWidth: Double): LocalGeometry? {
        if (segments.isEmpty()) return null
        val endpoints = mutableListOf<EndpointCandidate>()
        for (seg in segments.take(12)) {
            for ((seed, other) in listOf(seg.p1 to seg.p2, seg.p2 to seg.p1)) {
                if (!inside(mask, seed, 5)) continue
                val local = whiteFraction(mask, seed, max(10, (tableWidth * 0.020).toInt()))
                val center = whiteFraction(mask, seed, max(4, (tableWidth * 0.006).toInt()))
                // Hollow contact ring scores better than a filled white cue ball.
                val score = seg.length / tableWidth + local * 0.85 - center * 0.55
                endpoints += EndpointCandidate(seed, other, seg, score)
            }
        }

        var best: LocalGeometry? = null
        var bestScore = Double.NEGATIVE_INFINITY
        for (candidate in endpoints.sortedByDescending { it.score }.take(8)) {
            val roughIncoming = (candidate.seed - candidate.other).normalized()
            if (roughIncoming.length() < 0.5) continue
            val cropSide = max(108.0, tableWidth * 0.17).toInt().coerceAtMost(184)
            val pred = localDnn.predict(table, candidate.seed, cropSide) ?: continue
            val ring = extractRing(pred, candidate.seed, tableWidth) ?: continue
            if ((ring.center - candidate.seed).length() > cropSide * 0.29) continue

            val refinedIncoming = refineIncoming(mask, candidate.seg, ring.center, ring.radius, tableWidth)
            val incoming = refinedIncoming?.direction ?: (ring.center - candidate.other).normalized()
            val incomingSeg = refinedIncoming?.let { fit ->
                centeredGuideSegment(mask, candidate.seg, ring.center, fit, ring.radius)
            } ?: GuideSeg(candidate.other, ring.center, (ring.center - candidate.other).length())

            val branches = extractBranchComponents(pred, ring, incoming, tableWidth)
            if (branches.isEmpty()) continue
            val branchQ = branches.take(2).map { it.quality }.average()
            val incomingQ = refinedIncoming?.let { (1.0 / (1.0 + it.rms * 0.35)).coerceIn(0.0, 1.0) } ?: 0.45
            val geometryQ = (ring.quality * 0.35 + branchQ * 0.45 + incomingQ * 0.20).coerceIn(0.0, 1.0)
            val score = geometryQ * 100.0 + candidate.seg.length / tableWidth * 12.0 + if (branches.size >= 2) 3.0 else 0.0
            if (score > bestScore) {
                bestScore = score
                best = LocalGeometry(ring.center, incoming, incomingSeg, branches.take(2), ring.quality, geometryQ)
            }
        }
        return best
    }

    /** Connected component of class-2 pixels nearest the Hough endpoint, then circle-center refinement. */
    private fun extractRing(pred: LocalGuideDnn.Prediction, seed: Vec2, tableWidth: Double): RingFit? {
        val n = LocalGuideDnn.INPUT
        val visited = BooleanArray(n * n)
        var bestPoints: List<Vec2>? = null
        var bestScore = Double.NEGATIVE_INFINITY
        val qx = IntArray(n * n); val qy = IntArray(n * n)

        for (sy in 0 until n) for (sx in 0 until n) {
            val idx = sy * n + sx
            if (visited[idx] || pred.label(sx, sy) != 2) continue
            var head = 0; var tail = 0
            qx[tail] = sx; qy[tail] = sy; tail++; visited[idx] = true
            val points = mutableListOf<Vec2>()
            while (head < tail) {
                val x = qx[head]; val y = qy[head]; head++
                points += pred.tablePoint(x.toDouble(), y.toDouble())
                for (dy in -1..1) for (dx in -1..1) {
                    if (dx == 0 && dy == 0) continue
                    val nx = x + dx; val ny = y + dy
                    if (nx !in 0 until n || ny !in 0 until n) continue
                    val ni = ny * n + nx
                    if (!visited[ni] && pred.label(nx, ny) == 2) {
                        visited[ni] = true; qx[tail] = nx; qy[tail] = ny; tail++
                    }
                }
            }
            if (points.size < 6) continue
            val cx = points.sumOf { it.x } / points.size
            val cy = points.sumOf { it.y } / points.size
            val c = Vec2(cx, cy)
            val d = (c - seed).length()
            if (d > pred.side * 0.31) continue
            val score = points.size.toDouble() - d * 0.22
            if (score > bestScore) { bestScore = score; bestPoints = points }
        }
        val points = bestPoints ?: return null
        val cx0 = points.sumOf { it.x } / points.size
        val cy0 = points.sumOf { it.y } / points.size
        var bestCenter = Vec2(cx0, cy0)
        var bestRadius = 0.0
        var bestLoss = Double.POSITIVE_INFINITY
        val search = max(4, (tableWidth * 0.006).toInt()).coerceAtMost(8)
        for (dy in -search..search) for (dx in -search..search) {
            val c = Vec2(cx0 + dx, cy0 + dy)
            val radii = points.map { (it - c).length() }
            val mean = radii.average()
            if (mean < tableWidth * 0.007 || mean > tableWidth * 0.030) continue
            val variance = radii.sumOf { (it - mean) * (it - mean) } / radii.size
            val loss = variance + (c - seed).length() * 0.025
            if (loss < bestLoss) { bestLoss = loss; bestCenter = c; bestRadius = mean }
        }
        if (!bestLoss.isFinite() || bestRadius <= 0.0) return null
        val sizeQ = (points.size / 55.0).coerceIn(0.25, 1.0)
        val circleQ = exp(-sqrt(bestLoss.coerceAtLeast(0.0)) * 0.35).coerceIn(0.2, 1.0)
        return RingFit(bestCenter, bestRadius, (sizeQ * 0.45 + circleQ * 0.55).coerceIn(0.0, 1.0))
    }

    /** Re-fit the centerline, so Hough cannot leave us on one white edge of the native guide. */
    private fun refineIncoming(mask: Mat, seg: GuideSeg, junction: Vec2, ringRadius: Double, tableWidth: Double): Fit? {
        val other = if ((seg.p1 - junction).length() > (seg.p2 - junction).length()) seg.p1 else seg.p2
        val rough = (junction - other).normalized()
        if (rough.length() < 0.5) return null
        val corridor = max(3.0, tableWidth * 0.0048)
        val minX = max(0, (min(other.x, junction.x) - corridor - 3).toInt())
        val maxX = min(mask.cols() - 1, (max(other.x, junction.x) + corridor + 3).toInt())
        val minY = max(0, (min(other.y, junction.y) - corridor - 3).toInt())
        val maxY = min(mask.rows() - 1, (max(other.y, junction.y) + corridor + 3).toInt())
        val totalLen = (junction - other).length()
        val points = mutableListOf<Vec2>()
        for (y in minY..maxY) for (x in minX..maxX) {
            val v0 = mask.get(y, x) ?: continue
            if (v0.isEmpty() || v0[0] <= 0.0) continue
            val p = Vec2(x.toDouble(), y.toDouble())
            val v = p - other
            val along = v.dot(rough)
            if (along < 0.0 || along > totalLen - ringRadius * 1.10) continue
            val perp = abs(v.x * rough.y - v.y * rough.x)
            if (perp <= corridor) points += p
        }
        if (points.size < 24) return null
        return fitPoints(points, rough)
    }

    private fun centeredGuideSegment(mask: Mat, original: GuideSeg, junction: Vec2, fit: Fit, ringRadius: Double): GuideSeg {
        val other = if ((original.p1 - junction).length() > (original.p2 - junction).length()) original.p1 else original.p2
        val d = fit.direction
        val startT = (other - fit.center).dot(d)
        val endT = (junction - fit.center).dot(d) - ringRadius * 0.75
        val p1 = fit.center + d * startT
        val p2 = fit.center + d * endT
        return GuideSeg(p1, p2, (p2 - p1).length())
    }

    /**
     * Remove the contact disc and the incoming/backward side, then fit each remaining
     * connected class-1 component independently. No 2-degree ray quantization remains.
     */
    private fun extractBranchComponents(
        pred: LocalGuideDnn.Prediction,
        ring: RingFit,
        incoming: Vec2,
        tableWidth: Double
    ): List<RayBranch> {
        val n = LocalGuideDnn.INPUT
        val candidate = BooleanArray(n * n)
        val inner = max(ring.radius * 0.78, tableWidth * 0.010)
        val outer = min(pred.side * 0.49, max(48.0, tableWidth * 0.087))
        for (y in 0 until n) for (x in 0 until n) {
            if (pred.label(x, y) != 1) continue
            val p = pred.tablePoint(x.toDouble(), y.toDouble())
            val v = p - ring.center
            val radial = v.length()
            if (radial < inner || radial > outer) continue
            // Everything clearly behind the contact marker belongs to the incoming guide.
            if (v.dot(incoming) < -max(2.0, ring.radius * 0.10)) continue
            candidate[y * n + x] = true
        }

        val visited = BooleanArray(n * n)
        val qx = IntArray(n * n); val qy = IntArray(n * n)
        val result = mutableListOf<RayBranch>()
        for (sy in 0 until n) for (sx in 0 until n) {
            val si = sy * n + sx
            if (!candidate[si] || visited[si]) continue
            var head = 0; var tail = 0
            qx[tail] = sx; qy[tail] = sy; tail++; visited[si] = true
            val points = mutableListOf<Vec2>()
            while (head < tail) {
                val x = qx[head]; val y = qy[head]; head++
                points += pred.tablePoint(x.toDouble(), y.toDouble())
                for (dy in -1..1) for (dx in -1..1) {
                    if (dx == 0 && dy == 0) continue
                    val nx = x + dx; val ny = y + dy
                    if (nx !in 0 until n || ny !in 0 until n) continue
                    val ni = ny * n + nx
                    if (candidate[ni] && !visited[ni]) {
                        visited[ni] = true; qx[tail] = nx; qy[tail] = ny; tail++
                    }
                }
            }
            if (points.size < 7) continue
            val minR = points.minOf { (it - ring.center).length() }
            val maxR = points.maxOf { (it - ring.center).length() }
            if (minR > max(inner * 2.2, tableWidth * 0.035)) continue
            if (maxR - minR < max(9.0, tableWidth * 0.012)) continue

            val meanOut = Vec2(
                points.sumOf { it.x - ring.center.x } / points.size,
                points.sumOf { it.y - ring.center.y } / points.size
            ).normalized()
            val fit = fitPoints(points, meanOut) ?: continue
            if (fit.linearity < 4.0 || fit.rms > max(4.5, tableWidth * 0.0065)) continue
            var dir = fit.direction
            if (dir.dot(meanOut) < 0.0) dir = dir * -1.0
            if (dir.dot(incoming) < -0.16) continue

            val proj = points.map { (it - ring.center).dot(dir) }.filter { it > 0.0 }.sorted()
            if (proj.size < 7) continue
            val lo = quantile(proj, 0.08)
            val hi = quantile(proj, 0.97)
            val length = hi - lo
            if (length < max(9.0, tableWidth * 0.012)) continue
            val linearQ = ((fit.linearity - 3.0) / 12.0).coerceIn(0.2, 1.0)
            val residualQ = exp(-fit.rms * 0.28).coerceIn(0.15, 1.0)
            val lengthQ = (length / max(20.0, tableWidth * 0.035)).coerceIn(0.25, 1.0)
            val quality = (linearQ * 0.40 + residualQ * 0.35 + lengthQ * 0.25).coerceIn(0.0, 1.0)
            result += RayBranch(dir, ring.center + dir * lo, ring.center + dir * hi, quality * 100.0, length, quality)
        }

        val kept = mutableListOf<RayBranch>()
        for (b in result.sortedByDescending { it.score }) {
            if (kept.none { branchAngle(it.direction, b.direction) < 9.0 }) kept += b
            if (kept.size >= 3) break
        }
        return kept.take(2)
    }

    private fun fitPoints(points: List<Vec2>, desired: Vec2): Fit? {
        if (points.size < 3) return null
        val mx = points.sumOf { it.x } / points.size
        val my = points.sumOf { it.y } / points.size
        var xx = 0.0; var xy = 0.0; var yy = 0.0
        for (p in points) {
            val dx = p.x - mx; val dy = p.y - my
            xx += dx * dx; xy += dx * dy; yy += dy * dy
        }
        val trace = xx + yy
        val disc = sqrt(max(0.0, (xx - yy) * (xx - yy) + 4.0 * xy * xy))
        val l1 = (trace + disc) * 0.5
        val l2 = max(1e-6, (trace - disc) * 0.5)
        if (l1 <= 1e-6) return null
        val angle = 0.5 * atan2(2.0 * xy, xx - yy)
        var dir = Vec2(cos(angle), sin(angle)).normalized()
        if (dir.dot(desired) < 0.0) dir = dir * -1.0
        var residual = 0.0
        for (p in points) {
            val v = p - Vec2(mx, my)
            val d = abs(v.x * dir.y - v.y * dir.x)
            residual += d * d
        }
        return Fit(dir, Vec2(mx, my), sqrt(residual / points.size), l1 / l2)
    }

    /** Local Hough circle only around the collision marker, never a full-table ball detector. */
    private fun detectTargetBall(table: Mat, junction: Vec2, branches: List<RayBranch>, tableWidth: Double): TargetBall? {
        if (branches.isEmpty()) return null
        val search = max(62.0, tableWidth * 0.105).toInt()
        val x1 = max(0, junction.x.toInt() - search)
        val y1 = max(0, junction.y.toInt() - search)
        val x2 = min(table.cols(), junction.x.toInt() + search + 1)
        val y2 = min(table.rows(), junction.y.toInt() + search + 1)
        if (x2 - x1 < 40 || y2 - y1 < 40) return null
        val crop = table.submat(Rect(x1, y1, x2 - x1, y2 - y1))
        val gray = Mat()
        Imgproc.cvtColor(crop, gray, Imgproc.COLOR_RGBA2GRAY)
        crop.release()
        Imgproc.medianBlur(gray, gray, 5)
        val circles = Mat()
        val minR = max(7, (tableWidth * 0.009).toInt())
        val maxR = max(minR + 2, (tableWidth * 0.023).toInt())
        Imgproc.HoughCircles(
            gray, circles, Imgproc.HOUGH_GRADIENT, 1.2,
            max(16.0, tableWidth * 0.020), 105.0, 16.0, minR, maxR
        )
        gray.release()

        var best: TargetBall? = null
        var bestScore = Double.NEGATIVE_INFINITY
        val count = if (circles.rows() > 0) circles.cols() else 0
        for (i in 0 until count) {
            val c = circles.get(0, i) ?: continue
            if (c.size < 3) continue
            val center = Vec2(x1 + c[0], y1 + c[1])
            val radius = c[2]
            val v = center - junction
            val d = v.length()
            if (d < radius * 1.25 || d > radius * 3.35) continue // reject the contact ring and remote balls
            val dir = v.normalized()
            val align = branches.maxOf { it.direction.dot(dir) }
            if (align < 0.70) continue
            val diameterError = abs(d / max(1.0, radius * 2.0) - 1.0)
            val distanceQ = exp(-diameterError * 1.8)
            val alignQ = ((align - 0.70) / 0.30).coerceIn(0.0, 1.0)
            val quality = (alignQ * 0.62 + distanceQ * 0.38).coerceIn(0.0, 1.0)
            if (quality > bestScore) {
                bestScore = quality
                best = TargetBall(center, radius, quality)
            }
        }
        circles.release()
        return best
    }

    private fun classify(geometry: LocalGeometry, target: TargetBall?, tableWidth: Double): TypedFrame? {
        val branches = geometry.branches
        if (branches.isEmpty()) return null
        var objectBranch: RayBranch? = null
        var cueBranch: RayBranch? = null
        var targetQuality = 0.0

        if (target != null) {
            val targetDir = (target.center - geometry.junction).normalized()
            val ordered = branches.sortedByDescending { it.direction.dot(targetDir) }
            val best = ordered.first()
            val bestAlign = best.direction.dot(targetDir)
            if (bestAlign >= 0.78) {
                objectBranch = best
                targetQuality = target.quality * ((bestAlign - 0.72) / 0.28).coerceIn(0.25, 1.0)
                cueBranch = ordered.drop(1).firstOrNull { branchAngle(it.direction, best.direction) >= 9.0 }
            } else if (branches.size == 1 && bestAlign <= 0.48) {
                // Target ball is known, and the only native branch is clearly not along its center line.
                cueBranch = best
                targetQuality = target.quality * 0.72
            }
        }

        // If the target circle disappears for a single frame, preserve identity from recent stable geometry.
        if (objectBranch == null && cueBranch == null && history.isNotEmpty()) {
            val prev = history.peekLast()
            if ((prev.junction - geometry.junction).length() <= tableWidth * 0.018) {
                prev.objectBranch?.let { old ->
                    geometry.branches.minByOrNull { branchAngle(it.direction, old.direction) }?.let { b ->
                        if (branchAngle(b.direction, old.direction) <= 7.0) objectBranch = b
                    }
                }
                prev.cueBranch?.let { old ->
                    geometry.branches.filter { it !== objectBranch }.minByOrNull { branchAngle(it.direction, old.direction) }?.let { b ->
                        if (branchAngle(b.direction, old.direction) <= 7.0) cueBranch = b
                    }
                }
                if (objectBranch != null || cueBranch != null) targetQuality = 0.58
            }
        }

        // Deliberately do NOT restore the old "closest angle = object ball" rule.
        if (objectBranch == null && cueBranch == null) return null
        val branchQ = listOfNotNull(objectBranch, cueBranch).map { it.quality }.average()
        val quality = (geometry.geometryQuality * 0.48 + branchQ * 0.30 + targetQuality * 0.22).coerceIn(0.0, 1.0)
        return TypedFrame(
            geometry.junction, objectBranch, cueBranch, target?.center, quality,
            geometry.branches, geometry.incomingSeg
        )
    }

    private fun stabilize(current: TypedFrame, tableWidth: Double): TypedFrame? {
        if (history.isNotEmpty() && (history.peekLast().junction - current.junction).length() > tableWidth * 0.016) {
            history.clear()
        }
        history.addLast(current)
        while (history.size > 5) history.removeFirst()

        val objectStable = stabilizeKind(current.objectBranch, true, tableWidth)
        val cueStable = stabilizeKind(current.cueBranch, false, tableWidth)
        if (current.objectBranch != null && objectStable == null && current.cueBranch == null) return null
        if (current.cueBranch != null && cueStable == null && current.objectBranch == null) return null

        val qBoost = if (history.size >= 3) 0.06 else 0.0
        return current.copy(
            objectBranch = objectStable,
            cueBranch = cueStable,
            quality = (current.quality + qBoost).coerceAtMost(0.99)
        )
    }

    private fun stabilizeKind(current: RayBranch?, objectKind: Boolean, tableWidth: Double): RayBranch? {
        if (current == null) return null
        val candidates = mutableListOf<RayBranch>()
        for (f in history) {
            val b = if (objectKind) f.objectBranch else f.cueBranch
            if (b != null && branchAngle(b.direction, current.direction) <= 5.0) candidates += b
        }
        if (candidates.size <= 1) return current
        var sx = 0.0; var sy = 0.0
        for (b in candidates) { sx += b.direction.x; sy += b.direction.y }
        val dir = Vec2(sx, sy).normalized()
        val spread = candidates.maxOf { branchAngle(it.direction, dir) }
        // On a steady contact point a larger spread is measurement noise: hide rather than guess.
        if (history.size >= 3 && spread > 2.4) return null
        val end = Vec2(candidates.map { it.end.x }.average(), candidates.map { it.end.y }.average())
        val start = Vec2(candidates.map { it.start.x }.average(), candidates.map { it.start.y }.average())
        val length = candidates.map { it.length }.average()
        val quality = (candidates.map { it.quality }.average() * exp(-spread * 0.08)).coerceIn(0.0, 1.0)
        return RayBranch(dir, start, end, quality * 100.0, length, quality)
    }

    private fun whiteFraction(mask: Mat, p: Vec2, radius: Int): Double {
        val x1 = max(0, p.x.toInt() - radius); val y1 = max(0, p.y.toInt() - radius)
        val x2 = min(mask.cols(), p.x.toInt() + radius + 1); val y2 = min(mask.rows(), p.y.toInt() + radius + 1)
        if (x2 <= x1 || y2 <= y1) return 0.0
        val roi = mask.submat(Rect(x1, y1, x2 - x1, y2 - y1))
        val c = Core.countNonZero(roi); roi.release()
        return c.toDouble() / ((x2 - x1) * (y2 - y1)).toDouble()
    }

    private fun quantile(sorted: List<Double>, q: Double): Double {
        if (sorted.isEmpty()) return 0.0
        val i = (sorted.lastIndex * q).toInt().coerceIn(0, sorted.lastIndex)
        return sorted[i]
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
        // Calibrated to the inner playable cushion edge, not the outer ROI frame.
        val insetX = tableRect.width * 0.0145
        val insetY = tableRect.height * 0.021
        val bounds = PlayRect(insetX, insetY, tableRect.width - insetX, tableRect.height - insetY)
        var p = startLocal
        var d = direction0.normalized()
        if (d.length() < 0.5) return result
        for (i in 0..bounces) {
            val hit = rayToBounds(p, d, bounds) ?: break
            val kind = if (i == 0) firstKind else SegmentKind.BOUNCE
            result += TrajectorySegment(toGlobal(p, tableRect, scale), toGlobal(hit, tableRect, scale), kind)
            if (i == bounces) break
            var nx = d.x; var ny = d.y; val eps = 2.8
            if (abs(hit.x - bounds.left) < eps || abs(hit.x - bounds.right) < eps) nx = -nx
            if (abs(hit.y - bounds.top) < eps || abs(hit.y - bounds.bottom) < eps) ny = -ny
            d = Vec2(nx, ny).normalized(); p = hit + d * 1.5
        }
        return result
    }

    private fun rayToBounds(p: Vec2, d: Vec2, b: PlayRect): Vec2? {
        var best = Double.POSITIVE_INFINITY
        fun test(t: Double) {
            if (t <= 1e-6 || t >= best) return
            val x = p.x + d.x * t; val y = p.y + d.y * t
            if (x >= b.left - 0.5 && x <= b.right + 0.5 && y >= b.top - 0.5 && y <= b.bottom + 0.5) best = t
        }
        if (abs(d.x) > 1e-9) { test((b.left - p.x) / d.x); test((b.right - p.x) / d.x) }
        if (abs(d.y) > 1e-9) { test((b.top - p.y) / d.y); test((b.bottom - p.y) / d.y) }
        return if (best.isFinite()) p + d * best else null
    }

    private fun drawCross(out: MutableList<TrajectorySegment>, p: Vec2, rect: Rect, scale: Double, r: Double) {
        out += TrajectorySegment(toGlobal(Vec2(p.x - r, p.y), rect, scale), toGlobal(Vec2(p.x + r, p.y), rect, scale), SegmentKind.AIM)
        out += TrajectorySegment(toGlobal(Vec2(p.x, p.y - r), rect, scale), toGlobal(Vec2(p.x, p.y + r), rect, scale), SegmentKind.AIM)
    }

    private fun inside(mask: Mat, p: Vec2, margin: Int): Boolean =
        p.x >= margin && p.y >= margin && p.x < mask.cols() - margin && p.y < mask.rows() - margin

    private fun toGlobal(p: Vec2, tableRect: Rect, scale: Double): Vec2 =
        Vec2((tableRect.x + p.x) / scale, (tableRect.y + p.y) / scale)

    private fun branchAngle(a: Vec2, b: Vec2): Double {
        val na = a.normalized(); val nb = b.normalized()
        return Math.toDegrees(acos(na.dot(nb).coerceIn(-1.0, 1.0)))
    }
}