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
 * 4.2.1 keeps the full 4.2 pipeline on tables it already accepts.
 * If the old felt-colour gate rejects the frame, a second detector runs without
 * looking at felt hue at all: fixed UI geometry + white native guide + local ONNX ring.
 */
class NativeGuideMlAnalyzerV421(private val context: Context) {
    private val primary = NativeGuideMlAnalyzerV42(context)
    private val prefs = context.getSharedPreferences(MainActivity.PREFS, Context.MODE_PRIVATE)
    private val localDnn = LocalGuideDnn(context)

    private data class GuideSeg(val p1: Vec2, val p2: Vec2, val length: Double)
    private data class Ring(val center: Vec2, val radius: Double, val quality: Double)
    private data class Branch(
        val direction: Vec2,
        val start: Vec2,
        val end: Vec2,
        val length: Double,
        val score: Double
    )
    private data class Candidate(
        val junction: Vec2,
        val incoming: Vec2,
        val incomingSeg: GuideSeg,
        val branches: List<Branch>,
        val score: Double
    )
    private data class Fit(val direction: Vec2, val center: Vec2, val rms: Double, val linearity: Double)
    private data class TargetBall(val center: Vec2, val radius: Double, val quality: Double)

    fun analyze(bitmap: Bitmap): AnalysisResult {
        val normal = primary.analyze(bitmap)
        if (!normal.status.contains("игровой стол не виден")) {
            return normal.copy(status = normal.status.replace("ML 4.2", "ML 4.2.1"))
        }
        return analyzeWithoutFeltColour(bitmap)
    }

    private fun analyzeWithoutFeltColour(bitmap: Bitmap): AnalysisResult {
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)
        val scale = min(1.0, 1280.0 / src.cols().toDouble())
        val work = Mat()
        if (scale < 0.999) Imgproc.resize(src, work, Size(), scale, scale, Imgproc.INTER_AREA) else src.copyTo(work)
        src.release()

        val tableRect = fixedTable(work)
        if (tableRect == null) {
            work.release()
            return AnalysisResult(bitmap.width, bitmap.height, null, emptyList(), null, emptyList(), null, 0, "ML 4.2.1: кадр слишком мал")
        }

        val table = work.submat(tableRect)
        val playRect = PlayRect(
            tableRect.x / scale,
            tableRect.y / scale,
            (tableRect.x + tableRect.width) / scale,
            (tableRect.y + tableRect.height) / scale
        )
        val mask = buildWhiteMask(table)
        val longSegments = detectLongGuideSegments(mask, table.cols())
        val candidate = findCandidate(table, mask, longSegments, table.cols().toDouble())

        val out = mutableListOf<TrajectorySegment>()
        val debug = prefs.getBoolean(MainActivity.KEY_DEBUG, false)
        var status = "ML 4.2.1: ожидание прицеливания"
        var confidence = 20

        if (candidate != null) {
            val target = detectTargetBall(table, candidate.junction, candidate.branches, table.cols().toDouble())
            val typed = classifyBranches(candidate, target)
            val objectBranch = typed.first
            val cueBranch = typed.second

            objectBranch?.let { b ->
                val bounces = prefs.getInt(MainActivity.KEY_BOUNCES, 0).coerceIn(0, 2)
                out += extendBranch(b.end, b.direction, tableRect, scale, SegmentKind.OBJECT, bounces)
            }
            cueBranch?.let { b ->
                out += extendBranch(b.end, b.direction, tableRect, scale, SegmentKind.CUE_AFTER, 0)
            }

            if (debug) {
                out += TrajectorySegment(
                    toGlobal(candidate.incomingSeg.p1, tableRect, scale),
                    toGlobal(candidate.incomingSeg.p2, tableRect, scale),
                    SegmentKind.AIM
                )
                candidate.branches.forEach { b ->
                    out += TrajectorySegment(
                        toGlobal(b.start, tableRect, scale),
                        toGlobal(b.end, tableRect, scale),
                        SegmentKind.AIM
                    )
                }
                drawCross(out, candidate.junction, tableRect, scale, max(6.0, tableRect.width * 0.0075))
                target?.let { drawCross(out, it.center, tableRect, scale, max(5.0, tableRect.width * 0.006)) }
            }

            status = when {
                objectBranch != null && cueBranch != null && target != null -> "ML 4.2.1: объект + биток • без цвета стола"
                objectBranch != null && cueBranch != null -> "ML 4.2.1: две ветки • без цвета стола"
                objectBranch != null -> "ML 4.2.1: прицельный шар • без цвета стола"
                else -> "ML 4.2.1: белый шар • без цвета стола"
            }
            confidence = (candidate.score * 100.0).toInt().coerceIn(55, 97)
        } else if (longSegments.isNotEmpty()) {
            status = if (localDnn.lastError != null) "ML 4.2.1: модель недоступна" else "ML 4.2.1: контакт/ветки не подтверждены"
            confidence = 35
            if (debug) {
                longSegments.take(4).forEach { s ->
                    out += TrajectorySegment(toGlobal(s.p1, tableRect, scale), toGlobal(s.p2, tableRect, scale), SegmentKind.AIM)
                }
            }
        }

        mask.release(); table.release(); work.release()
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

    private fun buildWhiteMask(table: Mat): Mat {
        val rgb = Mat(); val hsv = Mat(); val mask = Mat()
        Imgproc.cvtColor(table, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
        rgb.release()
        Core.inRange(hsv, Scalar(0.0, 0.0, 148.0), Scalar(179.0, 86.0, 255.0), mask)
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
            val p1 = Vec2(l[0], l[1]); val p2 = Vec2(l[2], l[3])
            val len = (p2 - p1).length()
            if (len >= minLen) out += GuideSeg(p1, p2, len)
        }
        lines.release()
        return out.sortedByDescending { it.length }.take(16)
    }

    private fun findCandidate(table: Mat, mask: Mat, segments: List<GuideSeg>, width: Double): Candidate? {
        if (segments.isEmpty()) return null
        data class Endpoint(val seed: Vec2, val other: Vec2, val seg: GuideSeg, val score: Double)
        val endpoints = mutableListOf<Endpoint>()
        for (seg in segments.take(12)) {
            for ((seed, other) in listOf(seg.p1 to seg.p2, seg.p2 to seg.p1)) {
                val local = whiteFraction(mask, seed, max(10, (width * 0.020).toInt()))
                val center = whiteFraction(mask, seed, max(4, (width * 0.006).toInt()))
                endpoints += Endpoint(seed, other, seg, seg.length / width + local * 0.9 - center * 0.60)
            }
        }

        var best: Candidate? = null
        var bestValue = Double.NEGATIVE_INFINITY
        for (e in endpoints.sortedByDescending { it.score }.take(10)) {
            val cropSide = max(108.0, width * 0.17).toInt().coerceAtMost(184)
            val pred = localDnn.predict(table, e.seed, cropSide) ?: continue
            val ring = extractRing(pred, e.seed, width) ?: continue
            if ((ring.center - e.seed).length() > cropSide * 0.30) continue
            val incoming = (ring.center - e.other).normalized()
            if (incoming.length() < 0.5) continue
            val branches = scanAndRefineBranches(mask, ring, incoming, width)
            if (branches.isEmpty()) continue
            val q = (ring.quality * 0.38 + branches.take(2).map { (it.score / 100.0).coerceIn(0.0, 1.0) }.average() * 0.62).coerceIn(0.0, 1.0)
            val value = q * 100.0 + e.seg.length / width * 10.0 + if (branches.size >= 2) 3.0 else 0.0
            if (value > bestValue) {
                bestValue = value
                best = Candidate(ring.center, incoming, GuideSeg(e.other, ring.center, (ring.center - e.other).length()), branches.take(2), q)
            }
        }
        return best
    }

    private fun extractRing(pred: LocalGuideDnn.Prediction, seed: Vec2, width: Double): Ring? {
        val n = LocalGuideDnn.INPUT
        val visited = BooleanArray(n * n)
        val qx = IntArray(n * n); val qy = IntArray(n * n)
        var best: List<Vec2>? = null
        var bestScore = Double.NEGATIVE_INFINITY

        for (sy in 0 until n) for (sx in 0 until n) {
            val idx = sy * n + sx
            if (visited[idx] || pred.label(sx, sy) != 2) continue
            var head = 0; var tail = 0
            qx[tail] = sx; qy[tail] = sy; tail++; visited[idx] = true
            val pts = mutableListOf<Vec2>()
            while (head < tail) {
                val x = qx[head]; val y = qy[head]; head++
                pts += pred.tablePoint(x.toDouble(), y.toDouble())
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
            if (pts.size < 6) continue
            val c = Vec2(pts.sumOf { it.x } / pts.size, pts.sumOf { it.y } / pts.size)
            val d = (c - seed).length()
            if (d > pred.side * 0.31) continue
            val score = pts.size - d * 0.20
            if (score > bestScore) { bestScore = score; best = pts }
        }

        val pts = best ?: return null
        val center = Vec2(pts.sumOf { it.x } / pts.size, pts.sumOf { it.y } / pts.size)
        val radius = pts.map { (it - center).length() }.average()
        if (radius < width * 0.006 || radius > width * 0.032) return null
        val spread = sqrt(pts.sumOf { val d=(it-center).length()-radius; d*d } / pts.size)
        val qSize = (pts.size / 55.0).coerceIn(0.25, 1.0)
        val qCircle = exp(-spread * 0.35).coerceIn(0.2, 1.0)
        return Ring(center, radius, (qSize * 0.45 + qCircle * 0.55).coerceIn(0.0, 1.0))
    }

    private fun scanAndRefineBranches(mask: Mat, ring: Ring, incoming: Vec2, width: Double): List<Branch> {
        val inner = max(ring.radius * 0.80, width * 0.010)
        val outer = max(48.0, width * 0.082)
        val raw = mutableListOf<Branch>()
        var deg = 0
        while (deg < 360) {
            val a = Math.toRadians(deg.toDouble())
            val dir = Vec2(cos(a), sin(a))
            if (dir.dot(incoming) >= -0.10) traceRay(mask, ring.center, dir, inner, outer, width)?.let { raw += it }
            deg += 2
        }

        val coarse = mutableListOf<Branch>()
        for (b in raw.sortedByDescending { it.score }) {
            if (coarse.none { branchAngle(it.direction, b.direction) < 11.0 }) coarse += b
            if (coarse.size >= 4) break
        }

        val refined = coarse.mapNotNull { refineBranch(mask, ring.center, it, width, inner, outer) }
        val kept = mutableListOf<Branch>()
        for (b in refined.sortedByDescending { it.score }) {
            if (kept.none { branchAngle(it.direction, b.direction) < 9.0 }) kept += b
            if (kept.size >= 2) break
        }
        return kept
    }

    private fun traceRay(mask: Mat, junction: Vec2, dir: Vec2, inner: Double, outer: Double, width: Double): Branch? {
        val perp = Vec2(-dir.y, dir.x)
        var r = inner
        var started = false
        var first = 0.0
        var last = 0.0
        var hits = 0
        var samples = 0
        var missRun = 0
        var initialMiss = 0
        while (r <= outer) {
            val p = junction + dir * r
            val white = whiteAcross(mask, p, perp, max(2.5, width * 0.0038))
            samples++
            if (white) {
                if (!started) { started = true; first = r }
                last = r; hits++; missRun = 0
            } else if (!started) {
                initialMiss++
                if (initialMiss > 5) return null
            } else {
                missRun++
                if (missRun > 3) break
            }
            r += 1.5
        }
        if (!started || last - first < max(9.0, width * 0.012)) return null
        val ratio = hits.toDouble() / max(1, samples).toDouble()
        val len = last - first
        return Branch(dir, junction + dir * first, junction + dir * last, len, len + ratio * 22.0 - initialMiss * 1.2)
    }

    private fun refineBranch(mask: Mat, junction: Vec2, rough: Branch, width: Double, inner: Double, outer: Double): Branch? {
        val axis = rough.direction.normalized()
        val corridor = max(3.0, width * 0.0048)
        val points = mutableListOf<Vec2>()
        val x1 = max(0, (min(rough.start.x, rough.end.x) - corridor - 4).toInt())
        val x2 = min(mask.cols() - 1, (max(rough.start.x, rough.end.x) + corridor + 4).toInt())
        val y1 = max(0, (min(rough.start.y, rough.end.y) - corridor - 4).toInt())
        val y2 = min(mask.rows() - 1, (max(rough.start.y, rough.end.y) + corridor + 4).toInt())
        for (y in y1..y2) for (x in x1..x2) {
            val value = mask.get(y, x) ?: continue
            if (value.isEmpty() || value[0] <= 0.0) continue
            val p = Vec2(x.toDouble(), y.toDouble())
            val v = p - junction
            val radial = v.length()
            if (radial < inner * 0.75 || radial > outer * 1.05) continue
            val along = v.dot(axis)
            if (along <= 0.0) continue
            val perp = abs(v.x * axis.y - v.y * axis.x)
            if (perp <= corridor) points += p
        }
        if (points.size < 10) return rough
        val fit = fitPoints(points, axis) ?: return rough
        if (fit.linearity < 3.0 || fit.rms > max(4.0, width * 0.0055)) return rough
        var dir = fit.direction
        if (dir.dot(axis) < 0.0) dir = dir * -1.0
        val proj = points.map { (it - junction).dot(dir) }.filter { it > 0.0 }.sorted()
        if (proj.size < 10) return rough
        val lo = quantile(proj, 0.08)
        val hi = quantile(proj, 0.97)
        if (hi - lo < max(9.0, width * 0.012)) return rough
        val quality = (exp(-fit.rms * 0.25) * ((fit.linearity - 2.0) / 10.0).coerceIn(0.25, 1.0)).coerceIn(0.0, 1.0)
        return Branch(dir, junction + dir * lo, junction + dir * hi, hi - lo, quality * 100.0)
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

    private fun detectTargetBall(table: Mat, junction: Vec2, branches: List<Branch>, width: Double): TargetBall? {
        if (branches.isEmpty()) return null
        val search = max(62.0, width * 0.105).toInt()
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
        val minR = max(7, (width * 0.009).toInt())
        val maxR = max(minR + 2, (width * 0.023).toInt())
        Imgproc.HoughCircles(gray, circles, Imgproc.HOUGH_GRADIENT, 1.2, max(16.0, width * 0.020), 105.0, 15.0, minR, maxR)
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
            if (d < radius * 1.30 || d > radius * 3.30) continue
            val dir = v.normalized()
            val align = branches.maxOf { it.direction.dot(dir) }
            if (align < 0.72) continue
            val diameterError = abs(d / max(1.0, radius * 2.0) - 1.0)
            val q = (((align - 0.72) / 0.28).coerceIn(0.0, 1.0) * 0.62 + exp(-diameterError * 1.8) * 0.38).coerceIn(0.0, 1.0)
            if (q > bestScore) { bestScore = q; best = TargetBall(center, radius, q) }
        }
        circles.release()
        return best
    }

    private fun classifyBranches(candidate: Candidate, target: TargetBall?): Pair<Branch?, Branch?> {
        val branches = candidate.branches
        if (branches.isEmpty()) return null to null
        if (branches.size == 1) {
            val only = branches.first()
            if (target != null) {
                val align = only.direction.dot((target.center - candidate.junction).normalized())
                return if (align >= 0.78) only to null else null to only
            }
            return only to null
        }
        if (target != null) {
            val tdir = (target.center - candidate.junction).normalized()
            val ordered = branches.sortedByDescending { it.direction.dot(tdir) }
            val objectBranch = ordered.first()
            if (objectBranch.direction.dot(tdir) >= 0.76) return objectBranch to ordered[1]
        }
        // Only a provisional colour choice when the target circle itself is not confirmed.
        // Both geometric rays are still direct continuations of their native branches.
        val ordered = branches.sortedBy { branchAngle(candidate.incoming, it.direction) }
        return ordered.first() to ordered[1]
    }

    private fun whiteAcross(mask: Mat, p: Vec2, perp: Vec2, corridor: Double): Boolean {
        var hits = 0; var samples = 0
        var o = -corridor
        while (o <= corridor) {
            val q = p + perp * o
            val x = q.x.toInt(); val y = q.y.toInt()
            if (x in 0 until mask.cols() && y in 0 until mask.rows()) {
                samples++
                val v = mask.get(y, x)
                if (v != null && v.isNotEmpty() && v[0] > 0.0) hits++
            }
            o += 1.5
        }
        return hits >= max(2, samples / 4)
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
        return sorted[(sorted.lastIndex * q).toInt().coerceIn(0, sorted.lastIndex)]
    }

    private fun extendBranch(startLocal: Vec2, direction0: Vec2, tableRect: Rect, scale: Double, kind: SegmentKind, bounces: Int): List<TrajectorySegment> {
        val out = mutableListOf<TrajectorySegment>()
        val insetX = tableRect.width * 0.0145
        val insetY = tableRect.height * 0.021
        val bounds = PlayRect(insetX, insetY, tableRect.width - insetX, tableRect.height - insetY)
        var p = startLocal
        var d = direction0.normalized()
        if (d.length() < 0.5) return out
        for (i in 0..bounces) {
            val hit = rayToBounds(p, d, bounds) ?: break
            out += TrajectorySegment(toGlobal(p, tableRect, scale), toGlobal(hit, tableRect, scale), if (i == 0) kind else SegmentKind.BOUNCE)
            if (i == bounces) break
            var nx = d.x; var ny = d.y; val eps = 2.8
            if (abs(hit.x - bounds.left) < eps || abs(hit.x - bounds.right) < eps) nx = -nx
            if (abs(hit.y - bounds.top) < eps || abs(hit.y - bounds.bottom) < eps) ny = -ny
            d = Vec2(nx, ny).normalized(); p = hit + d * 1.5
        }
        return out
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
        out += TrajectorySegment(toGlobal(Vec2(p.x-r,p.y),rect,scale),toGlobal(Vec2(p.x+r,p.y),rect,scale),SegmentKind.AIM)
        out += TrajectorySegment(toGlobal(Vec2(p.x,p.y-r),rect,scale),toGlobal(Vec2(p.x,p.y+r),rect,scale),SegmentKind.AIM)
    }

    private fun toGlobal(p: Vec2, rect: Rect, scale: Double): Vec2 = Vec2((rect.x + p.x) / scale, (rect.y + p.y) / scale)

    private fun branchAngle(a: Vec2, b: Vec2): Double {
        val na = a.normalized(); val nb = b.normalized()
        return Math.toDegrees(acos(na.dot(nb).coerceIn(-1.0, 1.0)))
    }
}
