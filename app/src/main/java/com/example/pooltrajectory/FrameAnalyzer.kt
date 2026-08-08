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

    private var smoothedAim: Vec2? = null
    private var previousCue: Vec2? = null

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

                // Hollow diagnostic graphics have a felt-colored center; real balls do not.
                if (difference < 38.0) continue

                val ball = Ball(
                    Vec2(wx / scale, wy / scale),
                    c[2] / scale,
                    false
                )
                scored += ScoredBall(ball, difference)
            }
        }
        circles.release()

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
        var nativeGuideFound = false

        if (cue != null && labeled.size > 1) {
            val rawAim = detectNativeGuideAim(tableMat, cue, scale, table.rect)
            if (rawAim != null) {
                nativeGuideFound = true
                val aim = stabilizeAim(cue, orientAim(cue, rawAim, labeled))
                val solved = PhysicsEngine.solve(
                    cue,
                    aim,
                    labeled,
                    playRect,
                    prefs.getInt(MainActivity.KEY_BOUNCES, 1).coerceIn(0, 2)
                )
                ghost = solved.first
                segments = solved.second
            } else {
                smoothedAim = null
            }
        } else {
            smoothedAim = null
            previousCue = null
        }

        gray.release()
        tableMat.release()
        work.release()

        val confidence = when {
            cue == null -> 25
            !nativeGuideFound -> 45
            segments.isNotEmpty() -> 95
            else -> 65
        }
        val status = when {
            cue == null -> "биток не найден"
            !nativeGuideFound -> "штатная линия не найдена"
            segments.isEmpty() -> "ищу столкновение"
            else -> "штатная линия"
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

    /**
     * Detects the white aiming guideline already drawn by 8 Ball Pool.
     * We intentionally do not infer aim from the cue stick anymore.
     */
    private fun detectNativeGuideAim(
        tableRgba: Mat,
        cue: Ball,
        scale: Double,
        tableRect: Rect
    ): Vec2? {
        val rgb = Mat()
        val hsv = Mat()
        val whiteMask = Mat()
        Imgproc.cvtColor(tableRgba, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)

        // Native 8BP guide is bright, nearly achromatic white/grey.
        // Colored overlay paths, felt and most balls are rejected by low saturation.
        Core.inRange(
            hsv,
            Scalar(0.0, 0.0, 165.0),
            Scalar(179.0, 92.0, 255.0),
            whiteMask
        )

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
        Imgproc.morphologyEx(whiteMask, whiteMask, Imgproc.MORPH_CLOSE, kernel)
        kernel.release()

        val cueLocalX = cue.center.x * scale - tableRect.x
        val cueLocalY = cue.center.y * scale - tableRect.y
        val cueRadius = cue.radius * scale

        val lines = Mat()
        val minLength = max(28.0, cueRadius * 3.2)
        Imgproc.HoughLinesP(
            whiteMask,
            lines,
            1.0,
            Math.PI / 360.0,
            22,
            minLength,
            max(10.0, cueRadius * 1.2)
        )

        var bestDirection: Vec2? = null
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
            if (len < minLength) continue

            val distance = distancePointToSegment(
                cueLocalX,
                cueLocalY,
                x1,
                y1,
                x2,
                y2
            )
            if (distance > cueRadius * 2.5) continue

            val base = Vec2(dx, dy).normalized()
            if (base.length() < 0.5) continue

            val scoreForward = guideContinuityScore(
                whiteMask,
                cueLocalX,
                cueLocalY,
                base,
                cueRadius
            )
            val reverse = base * -1.0
            val scoreReverse = guideContinuityScore(
                whiteMask,
                cueLocalX,
                cueLocalY,
                reverse,
                cueRadius
            )

            val direction: Vec2
            val continuity: Double
            if (scoreForward >= scoreReverse) {
                direction = base
                continuity = scoreForward
            } else {
                direction = reverse
                continuity = scoreReverse
            }

            // The native guide is long and continuous. Circular white regions on striped
            // balls may produce Hough fragments, but they score poorly on ray continuity.
            if (continuity < 0.22) continue
            val score = continuity * 600.0 + len - distance * 5.0
            if (score > bestScore) {
                bestScore = score
                bestDirection = direction
            }
        }

        lines.release()
        whiteMask.release()
        hsv.release()
        rgb.release()
        return bestDirection
    }

    private fun guideContinuityScore(
        mask: Mat,
        cx: Double,
        cy: Double,
        direction: Vec2,
        cueRadius: Double
    ): Double {
        val d = direction.normalized()
        val normal = Vec2(-d.y, d.x)
        val start = max(2.0, cueRadius * 1.15)
        val end = max(start + 20.0, cueRadius * 24.0)
        val step = 2.0

        var hits = 0
        var samples = 0
        var t = start
        while (t <= end) {
            val px = cx + d.x * t
            val py = cy + d.y * t
            if (px < 1.0 || py < 1.0 || px >= mask.cols() - 1.0 || py >= mask.rows() - 1.0) break

            var found = false
            for (offset in -2..2) {
                val sx = (px + normal.x * offset).toInt().coerceIn(0, mask.cols() - 1)
                val sy = (py + normal.y * offset).toInt().coerceIn(0, mask.rows() - 1)
                val value = mask.get(sy, sx)
                if (value != null && value.isNotEmpty() && value[0] > 0.0) {
                    found = true
                    break
                }
            }
            if (found) hits++
            samples++
            t += step
        }
        return if (samples == 0) 0.0 else hits.toDouble() / samples
    }

    private fun distancePointToSegment(
        px: Double,
        py: Double,
        x1: Double,
        y1: Double,
        x2: Double,
        y2: Double
    ): Double {
        val dx = x2 - x1
        val dy = y2 - y1
        val len2 = dx * dx + dy * dy
        if (len2 < 1e-9) return hypot(px - x1, py - y1)
        val t = (((px - x1) * dx + (py - y1) * dy) / len2).coerceIn(0.0, 1.0)
        return hypot(px - (x1 + t * dx), py - (y1 + t * dy))
    }

    private fun stabilizeAim(cue: Ball, raw: Vec2): Vec2 {
        val oldCue = previousCue
        val oldAim = smoothedAim
        previousCue = cue.center

        if (oldCue == null || oldAim == null || (cue.center - oldCue).length() > cue.radius * 1.8) {
            smoothedAim = raw.normalized()
            return smoothedAim!!
        }

        val blended = (oldAim * 0.72 + raw.normalized() * 0.28).normalized()
        smoothedAim = blended
        return blended
    }

    private fun detectTable(work: Mat): TableInfo {
        val rgb = Mat()
        val hsv = Mat()
        val mask = Mat()
        Imgproc.cvtColor(work, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)

        // Covers the common green, cyan and blue 8 Ball Pool felt skins.
        Core.inRange(hsv, Scalar(25.0, 32.0, 28.0), Scalar(118.0, 255.0, 255.0), mask)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(13.0, 13.0))
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel)
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel)

        val mean = Core.mean(work, mask)
        val felt = doubleArrayOf(mean.`val`[0], mean.`val`[1], mean.`val`[2], 255.0)

        val contourMask = mask.clone()
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(contourMask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        val minArea = work.cols().toDouble() * work.rows().toDouble() * 0.12
        val best = contours
            .filter { Imgproc.contourArea(it) >= minArea }
            .maxByOrNull { Imgproc.contourArea(it) }

        val detected = best != null
        var rect = if (best != null) {
            Imgproc.boundingRect(best)
        } else if (work.cols() > work.rows()) {
            // 8BP landscape layout profile, matched to the realme 16 Pro+ capture.
            Rect(
                (work.cols() * 0.18).toInt(),
                (work.rows() * 0.22).toInt(),
                (work.cols() * 0.64).toInt(),
                (work.rows() * 0.69).toInt()
            )
        } else {
            Rect(
                (work.cols() * 0.08).toInt(),
                (work.rows() * 0.18).toInt(),
                (work.cols() * 0.84).toInt(),
                (work.rows() * 0.64).toInt()
            )
        }

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
                        if (hi >= 178 && hi - lo <= 72) white++
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
