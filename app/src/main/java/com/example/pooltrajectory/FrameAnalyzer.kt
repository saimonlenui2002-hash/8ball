package com.example.pooltrajectory

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class FrameAnalyzer(private val context: Context) {
    private val prefs = context.getSharedPreferences(MainActivity.PREFS, Context.MODE_PRIVATE)

    private data class TableInfo(val rect: Rect, val felt: DoubleArray, val detected: Boolean)
    private data class ScoredBall(val ball: Ball, val score: Double)

    fun analyze(bitmap: Bitmap): AnalysisResult {
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)
        val scale = min(1.0, 960.0 / src.cols().toDouble())
        val work = Mat()
        if (scale < 0.999) {
            Imgproc.resize(src, work, Size(), scale, scale, Imgproc.INTER_AREA)
        } else {
            src.copyTo(work)
        }
        src.release()

        val table = detectTable(work)
        val tableMat = work.submat(table.rect)
        val gray = Mat()
        Imgproc.cvtColor(tableMat, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(7.0, 7.0), 1.4)

        val sensitivity = prefs.getInt(MainActivity.KEY_SENSITIVITY, 22).coerceIn(12, 38)
        val minR = max(7, (table.rect.width * 0.010).toInt())
        val maxR = max(minR + 3, (table.rect.width * 0.023).toInt())
        val circles = Mat()
        Imgproc.HoughCircles(
            gray,
            circles,
            Imgproc.HOUGH_GRADIENT,
            1.2,
            minR * 2.0,
            110.0,
            sensitivity.toDouble(),
            minR,
            maxR
        )

        val scored = mutableListOf<ScoredBall>()
        if (circles.rows() > 0) {
            for (i in 0 until circles.cols()) {
                val c = circles.get(0, i) ?: continue
                if (c.size < 3) continue
                val wx = c[0] + table.rect.x
                val wy = c[1] + table.rect.y
                val ix = wx.toInt().coerceIn(0, work.cols() - 1)
                val iy = wy.toInt().coerceIn(0, work.rows() - 1)
                val centerColor = work.get(iy, ix) ?: continue
                val difference = colorDistance(centerColor, table.felt)

                // Our diagnostic circles are hollow: their centers remain the felt color.
                // Real pool balls strongly differ from the felt at the detected center.
                if (difference < 42.0) continue

                val ball = Ball(
                    Vec2(wx / scale, wy / scale),
                    c[2] / scale,
                    false
                )
                scored += ScoredBall(ball, difference)
            }
        }
        circles.release()

        // A pool table can contain at most 16 balls. Keeping only the strongest
        // candidates prevents UI decorations and captured overlay graphics from flooding the view.
        val balls = scored
            .sortedByDescending { it.score }
            .take(16)
            .map { it.ball }

        val cueIndex = findCueBall(bitmap, balls)
        val labeled = balls.mapIndexed { index, ball -> ball.copy(isCue = index == cueIndex) }
        val cue = labeled.getOrNull(cueIndex)
        val playRect = PlayRect(
            table.rect.x / scale,
            table.rect.y / scale,
            (table.rect.x + table.rect.width) / scale,
            (table.rect.y + table.rect.height) / scale
        )

        var ghost: Vec2? = null
        var segments = emptyList<TrajectorySegment>()

        if (cue != null && labeled.size > 1) {
            val rawAim = detectAim(tableMat, cue, scale, table.rect, table.felt)
            if (rawAim != null) {
                val aim = orientAim(cue, rawAim, labeled)
                val solved = PhysicsEngine.solve(
                    cue,
                    aim,
                    labeled,
                    playRect,
                    prefs.getInt(MainActivity.KEY_BOUNCES, 1).coerceIn(0, 2)
                )
                ghost = solved.first
                segments = solved.second
            }
        }

        gray.release()
        tableMat.release()
        work.release()

        val confidence = when {
            !table.detected -> 35
            cue == null -> 30
            labeled.size in 2..16 && segments.isNotEmpty() -> 90
            labeled.size in 2..16 -> 70
            else -> 45
        }
        val status = when {
            !table.detected -> "ищу стол"
            cue == null -> "биток не найден"
            segments.isEmpty() -> "наведи кий"
            else -> "траектория"
        }
        return AnalysisResult(
            bitmap.width,
            bitmap.height,
            cue,
            labeled,
            ghost,
            segments,
            playRect,
            confidence,
            status
        )
    }

    private fun detectTable(work: Mat): TableInfo {
        val rgb = Mat()
        val hsv = Mat()
        val mask = Mat()
        Imgproc.cvtColor(work, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)

        // The current game skin uses green felt. This broad range tolerates shadows,
        // highlights and compression while rejecting the surrounding UI and rails.
        Core.inRange(hsv, Scalar(28.0, 45.0, 30.0), Scalar(100.0, 255.0, 255.0), mask)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(13.0, 13.0))
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel)
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel)

        val mean = Core.mean(work, mask)
        val felt = doubleArrayOf(mean.`val`[0], mean.`val`[1], mean.`val`[2], 255.0)

        val contourMask = mask.clone()
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(contourMask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        val minArea = work.cols().toDouble() * work.rows().toDouble() * 0.15
        val best = contours
            .filter { Imgproc.contourArea(it) >= minArea }
            .maxByOrNull { Imgproc.contourArea(it) }

        val detected = best != null
        var rect = if (best != null) {
            Imgproc.boundingRect(best)
        } else {
            Rect(
                (work.cols() * 0.15).toInt(),
                (work.rows() * 0.17).toInt(),
                (work.cols() * 0.70).toInt(),
                (work.rows() * 0.70).toInt()
            )
        }

        // Remove a thin cushion/rail edge from the analysis area.
        val padX = max(2, (rect.width * 0.008).toInt())
        val padY = max(2, (rect.height * 0.010).toInt())
        if (rect.width > padX * 2 + 20 && rect.height > padY * 2 + 20) {
            rect = Rect(rect.x + padX, rect.y + padY, rect.width - padX * 2, rect.height - padY * 2)
        }
        rect.x = rect.x.coerceIn(0, work.cols() - 1)
        rect.y = rect.y.coerceIn(0, work.rows() - 1)
        rect.width = rect.width.coerceIn(1, work.cols() - rect.x)
        rect.height = rect.height.coerceIn(1, work.rows() - rect.y)

        contours.forEach { it.release() }
        hierarchy.release()
        contourMask.release()
        kernel.release()
        mask.release()
        hsv.release()
        rgb.release()
        return TableInfo(rect, felt, detected)
    }

    private fun colorDistance(color: DoubleArray, felt: DoubleArray): Double {
        if (color.size < 3 || felt.size < 3) return 0.0
        val dr = color[0] - felt[0]
        val dg = color[1] - felt[1]
        val db = color[2] - felt[2]
        return sqrt(dr * dr + dg * dg + db * db)
    }

    private fun findCueBall(bitmap: Bitmap, balls: List<Ball>): Int {
        if (balls.isEmpty()) return -1
        var bestIndex = -1
        var bestScore = Double.NEGATIVE_INFINITY

        balls.forEachIndexed { index, ball ->
            val cx = ball.center.x.toInt().coerceIn(0, bitmap.width - 1)
            val cy = ball.center.y.toInt().coerceIn(0, bitmap.height - 1)
            val radius = max(3, (ball.radius * 0.62).toInt())
            val step = max(1, radius / 4)
            var white = 0
            var total = 0
            var luminance = 0.0

            var dy = -radius
            while (dy <= radius) {
                var dx = -radius
                while (dx <= radius) {
                    if (dx * dx + dy * dy <= radius * radius) {
                        val x = (cx + dx).coerceIn(0, bitmap.width - 1)
                        val y = (cy + dy).coerceIn(0, bitmap.height - 1)
                        val pixel = bitmap.getPixel(x, y)
                        val r = Color.red(pixel)
                        val g = Color.green(pixel)
                        val b = Color.blue(pixel)
                        val hi = maxOf(r, g, b)
                        val lo = minOf(r, g, b)
                        if (hi >= 180 && hi - lo <= 70) white++
                        luminance += (r + g + b) / 3.0
                        total++
                    }
                    dx += step
                }
                dy += step
            }

            if (total > 0) {
                val whiteFraction = white.toDouble() / total
                val score = whiteFraction * 1000.0 + luminance / total
                if (score > bestScore) {
                    bestScore = score
                    bestIndex = index
                }
            }
        }
        return bestIndex
    }

    private fun detectAim(
        tableRgba: Mat,
        cue: Ball,
        scale: Double,
        tableRect: Rect,
        felt: DoubleArray
    ): Vec2? {
        val clean = tableRgba.clone()
        eraseOverlayColor(clean, 70, 255, 110, felt)
        eraseOverlayColor(clean, 255, 210, 55, felt)
        eraseOverlayColor(clean, 70, 210, 255, felt)
        eraseOverlayColor(clean, 255, 170, 60, felt)

        val gray = Mat()
        Imgproc.cvtColor(clean, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 1.1)
        val edges = Mat()
        Imgproc.Canny(gray, edges, 55.0, 145.0)
        val lines = Mat()
        Imgproc.HoughLinesP(edges, lines, 1.0, Math.PI / 180.0, 38, 45.0, 14.0)

        val cx = cue.center.x * scale - tableRect.x
        val cy = cue.center.y * scale - tableRect.y
        val cueRadius = cue.radius * scale
        var best: Vec2? = null
        var bestScore = Double.NEGATIVE_INFINITY

        for (row in 0 until lines.rows()) {
            val l = lines.get(row, 0) ?: continue
            if (l.size < 4) continue
            val x1 = l[0]
            val y1 = l[1]
            val x2 = l[2]
            val y2 = l[3]
            val dx = x2 - x1
            val dy = y2 - y1
            val len = hypot(dx, dy)
            if (len < 38.0) continue

            val t = ((cx - x1) * dx + (cy - y1) * dy) / (len * len)
            val clamped = t.coerceIn(0.0, 1.0)
            val px = x1 + clamped * dx
            val py = y1 + clamped * dy
            val distance = hypot(px - cx, py - cy)
            if (distance > cueRadius * 2.4) continue

            val d1 = hypot(x1 - cx, y1 - cy)
            val d2 = hypot(x2 - cx, y2 - cy)
            val farX = if (d1 >= d2) x1 else x2
            val farY = if (d1 >= d2) y1 else y2
            val direction = Vec2(farX - cx, farY - cy).normalized()
            if (direction.length() < 0.5) continue

            val score = len / (1.0 + distance * 2.2)
            if (score > bestScore) {
                bestScore = score
                best = direction
            }
        }

        lines.release()
        edges.release()
        gray.release()
        clean.release()
        return best
    }

    private fun eraseOverlayColor(image: Mat, r: Int, g: Int, b: Int, felt: DoubleArray) {
        val tolerance = 16
        val mask = Mat()
        Core.inRange(
            image,
            Scalar(
                max(0, r - tolerance).toDouble(),
                max(0, g - tolerance).toDouble(),
                max(0, b - tolerance).toDouble(),
                0.0
            ),
            Scalar(
                min(255, r + tolerance).toDouble(),
                min(255, g + tolerance).toDouble(),
                min(255, b + tolerance).toDouble(),
                255.0
            ),
            mask
        )
        image.setTo(Scalar(felt[0], felt[1], felt[2], 255.0), mask)
        mask.release()
    }

    private fun orientAim(cue: Ball, raw: Vec2, balls: List<Ball>): Vec2 {
        val forward = PhysicsEngine.nearestBallCollision(cue, raw, balls)
        val backwardDirection = raw * -1.0
        val backward = PhysicsEngine.nearestBallCollision(cue, backwardDirection, balls)
        return when {
            forward != null && backward == null -> raw
            backward != null && forward == null -> backwardDirection
            forward != null && backward != null -> if (forward.t <= backward.t) raw else backwardDirection
            else -> raw
        }
    }
}
