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

class GuideEndpointAnalyzer(private val context: Context) {
    private val prefs = context.getSharedPreferences(MainActivity.PREFS, Context.MODE_PRIVATE)

    private data class TableInfo(val rect: Rect, val felt: DoubleArray, val detected: Boolean)
    private data class GuideLine(val p1: Vec2, val p2: Vec2, val length: Double)
    private data class BallCandidate(val ball: Ball, val contrast: Double)

    private var smoothedAim: Vec2? = null
    private var previousCue: Vec2? = null

    fun analyze(bitmap: Bitmap): AnalysisResult {
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)
        val scale = min(1.0, 960.0 / src.cols().toDouble())
        val work = Mat()
        if (scale < 0.999) Imgproc.resize(src, work, Size(), scale, scale, Imgproc.INTER_AREA) else src.copyTo(work)
        src.release()

        val table = detectTable(work)
        val tableMat = work.submat(table.rect)
        val guideLines = detectGuideLines(tableMat)
        val balls = detectBalls(bitmap, work, tableMat, table, scale)

        val cueIndex = chooseCueBall(bitmap, balls, guideLines, scale, table.rect)
        val labeled = balls.mapIndexed { i, b -> b.copy(isCue = i == cueIndex) }
        val cue = labeled.getOrNull(cueIndex)

        val playRect = PlayRect(
            table.rect.x / scale,
            table.rect.y / scale,
            (table.rect.x + table.rect.width) / scale,
            (table.rect.y + table.rect.height) / scale
        )

        var ghost: Vec2? = null
        var segments = emptyList<TrajectorySegment>()
        var lineFound = false

        if (cue != null) {
            val rawAim = chooseGuideAim(cue, guideLines, scale, table.rect)
            if (rawAim != null) {
                lineFound = true
                val aim = stabilize(cue, rawAim)
                val solved = PhysicsEngine.solve(
                    cue,
                    aim,
                    labeled,
                    playRect,
                    prefs.getInt(MainActivity.KEY_BOUNCES, 1).coerceIn(0, 2)
                )
                ghost = solved.first
                // The game already draws the cue-to-contact segment. Only draw what comes after it.
                segments = solved.second.filter { it.kind != SegmentKind.AIM }
            } else {
                smoothedAim = null
            }
        } else {
            smoothedAim = null
            previousCue = null
        }

        tableMat.release()
        work.release()

        val status = when {
            cue == null -> "биток по штатной линии не найден"
            !lineFound -> "штатная линия не найдена"
            ghost == null -> "нет столкновения"
            else -> "продолжение штатной линии"
        }
        val confidence = when {
            cue == null -> 20
            !lineFound -> 35
            ghost == null -> 60
            else -> 95
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

    private fun detectGuideLines(tableRgba: Mat): List<GuideLine> {
        val rgb = Mat()
        val hsv = Mat()
        val mask = Mat()
        Imgproc.cvtColor(tableRgba, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)

        // Native 8BP guideline: bright and nearly achromatic.
        Core.inRange(hsv, Scalar(0.0, 0.0, 155.0), Scalar(179.0, 72.0, 255.0), mask)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel)
        kernel.release()

        val lines = Mat()
        val minLength = max(35.0, tableRgba.cols() * 0.13)
        Imgproc.HoughLinesP(mask, lines, 1.0, Math.PI / 360.0, 20, minLength, 12.0)

        val out = mutableListOf<GuideLine>()
        val edgePadX = tableRgba.cols() * 0.025
        val edgePadY = tableRgba.rows() * 0.035
        for (row in 0 until lines.rows()) {
            val l = lines.get(row, 0) ?: continue
            if (l.size < 4) continue
            val p1 = Vec2(l[0], l[1])
            val p2 = Vec2(l[2], l[3])
            val len = (p2 - p1).length()
            if (len < minLength) continue

            val mx = (p1.x + p2.x) * 0.5
            val my = (p1.y + p2.y) * 0.5
            val nearBorder = mx < edgePadX || mx > tableRgba.cols() - edgePadX || my < edgePadY || my > tableRgba.rows() - edgePadY
            if (nearBorder) continue
            out += GuideLine(p1, p2, len)
        }

        lines.release()
        mask.release()
        hsv.release()
        rgb.release()
        return out.sortedByDescending { it.length }.take(24)
    }

    private fun detectBalls(
        bitmap: Bitmap,
        work: Mat,
        tableRgba: Mat,
        table: TableInfo,
        scale: Double
    ): List<Ball> {
        val gray = Mat()
        Imgproc.cvtColor(tableRgba, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(7.0, 7.0), 1.4)

        val sensitivity = prefs.getInt(MainActivity.KEY_SENSITIVITY, 22).coerceIn(12, 38)
        val minR = max(6, (table.rect.width * 0.009).toInt())
        val maxR = max(minR + 3, (table.rect.width * 0.0225).toInt())
        val circles = Mat()
        Imgproc.HoughCircles(gray, circles, Imgproc.HOUGH_GRADIENT, 1.15, minR * 1.8, 105.0, sensitivity.toDouble(), minR, maxR)

        val candidates = mutableListOf<BallCandidate>()
        if (circles.rows() > 0) {
            for (i in 0 until circles.cols()) {
                val c = circles.get(0, i) ?: continue
                if (c.size < 3) continue
                val wx = c[0] + table.rect.x
                val wy = c[1] + table.rect.y
                val ix = wx.toInt().coerceIn(0, work.cols() - 1)
                val iy = wy.toInt().coerceIn(0, work.rows() - 1)
                val center = work.get(iy, ix) ?: continue
                val contrast = colorDistance(center, table.felt)
                if (contrast < 34.0) continue

                val ball = Ball(Vec2(wx / scale, wy / scale), c[2] / scale, false)
                // Reject obviously hollow graphics: real balls have non-felt pixels through the interior.
                if (interiorNonFeltFraction(bitmap, ball, table.felt) < 0.38) continue
                candidates += BallCandidate(ball, contrast)
            }
        }

        circles.release()
        gray.release()
        return candidates.sortedByDescending { it.contrast }.take(16).map { it.ball }
    }

    private fun chooseCueBall(
        bitmap: Bitmap,
        balls: List<Ball>,
        lines: List<GuideLine>,
        scale: Double,
        tableRect: Rect
    ): Int {
        if (balls.isEmpty() || lines.isEmpty()) return -1

        var bestIndex = -1
        var bestScore = Double.NEGATIVE_INFINITY
        balls.forEachIndexed { index, ball ->
            val whiteScore = cueWhitenessScore(bitmap, ball)
            if (whiteScore < 0.25) return@forEachIndexed

            val local = Vec2(ball.center.x * scale - tableRect.x, ball.center.y * scale - tableRect.y)
            val radius = ball.radius * scale
            var guideScore = Double.NEGATIVE_INFINITY

            for (line in lines) {
                val dist = distancePointToLine(local, line.p1, line.p2)
                if (dist > radius * 2.2) continue
                val endDistance = min((local - line.p1).length(), (local - line.p2).length())
                if (endDistance > radius * 9.0) continue

                val endpoint = (1.0 - endDistance / (radius * 9.0)).coerceIn(0.0, 1.0)
                val lineStrength = (line.length / max(1.0, tableRect.width.toDouble())).coerceAtMost(1.0)
                val score = endpoint * 5.0 + lineStrength * 2.0 - dist / max(1.0, radius)
                if (score > guideScore) guideScore = score
            }

            if (!guideScore.isFinite()) return@forEachIndexed
            val score = whiteScore * 7.0 + guideScore
            if (score > bestScore) {
                bestScore = score
                bestIndex = index
            }
        }
        return bestIndex
    }

    private fun chooseGuideAim(
        cue: Ball,
        lines: List<GuideLine>,
        scale: Double,
        tableRect: Rect
    ): Vec2? {
        val local = Vec2(cue.center.x * scale - tableRect.x, cue.center.y * scale - tableRect.y)
        val radius = cue.radius * scale
        var best: GuideLine? = null
        var bestScore = Double.NEGATIVE_INFINITY

        for (line in lines) {
            val dist = distancePointToLine(local, line.p1, line.p2)
            if (dist > radius * 2.2) continue
            val d1 = (local - line.p1).length()
            val d2 = (local - line.p2).length()
            val near = min(d1, d2)
            if (near > radius * 9.0) continue
            val score = line.length - dist * 5.0 - near * 0.5
            if (score > bestScore) {
                bestScore = score
                best = line
            }
        }

        val line = best ?: return null
        val d1 = (local - line.p1).length()
        val d2 = (local - line.p2).length()
        val far = if (d1 >= d2) line.p1 else line.p2
        val direction = (far - local).normalized()
        return if (direction.length() > 0.5) direction else null
    }

    private fun cueWhitenessScore(bitmap: Bitmap, ball: Ball): Double {
        val cx = ball.center.x.toInt().coerceIn(0, bitmap.width - 1)
        val cy = ball.center.y.toInt().coerceIn(0, bitmap.height - 1)
        val radius = max(4, (ball.radius * 0.72).toInt())
        val step = max(1, radius / 6)
        val hsv = FloatArray(3)
        var white = 0
        var colored = 0
        var dark = 0
        var total = 0

        var dy = -radius
        while (dy <= radius) {
            var dx = -radius
            while (dx <= radius) {
                if (dx * dx + dy * dy <= radius * radius) {
                    val x = (cx + dx).coerceIn(0, bitmap.width - 1)
                    val y = (cy + dy).coerceIn(0, bitmap.height - 1)
                    val p = bitmap.getPixel(x, y)
                    Color.RGBToHSV(Color.red(p), Color.green(p), Color.blue(p), hsv)
                    val s = hsv[1]
                    val v = hsv[2]
                    if (v > 0.66f && s < 0.30f) white++
                    if (v > 0.35f && s > 0.38f) colored++
                    if (v < 0.28f) dark++
                    total++
                }
                dx += step
            }
            dy += step
        }
        if (total == 0) return -1.0
        val wf = white.toDouble() / total
        val cf = colored.toDouble() / total
        val df = dark.toDouble() / total
        return wf * 2.4 - cf * 1.7 - df * 0.5
    }

    private fun interiorNonFeltFraction(bitmap: Bitmap, ball: Ball, felt: DoubleArray): Double {
        val cx = ball.center.x.toInt().coerceIn(0, bitmap.width - 1)
        val cy = ball.center.y.toInt().coerceIn(0, bitmap.height - 1)
        val radius = max(3, (ball.radius * 0.62).toInt())
        val step = max(1, radius / 5)
        var nonFelt = 0
        var total = 0

        var dy = -radius
        while (dy <= radius) {
            var dx = -radius
            while (dx <= radius) {
                if (dx * dx + dy * dy <= radius * radius) {
                    val x = (cx + dx).coerceIn(0, bitmap.width - 1)
                    val y = (cy + dy).coerceIn(0, bitmap.height - 1)
                    val p = bitmap.getPixel(x, y)
                    val c = doubleArrayOf(Color.red(p).toDouble(), Color.green(p).toDouble(), Color.blue(p).toDouble())
                    if (colorDistance(c, felt) > 32.0) nonFelt++
                    total++
                }
                dx += step
            }
            dy += step
        }
        return if (total == 0) 0.0 else nonFelt.toDouble() / total
    }

    private fun stabilize(cue: Ball, raw: Vec2): Vec2 {
        val oldCue = previousCue
        val old = smoothedAim
        previousCue = cue.center
        if (oldCue == null || old == null || (cue.center - oldCue).length() > cue.radius * 1.8) {
            smoothedAim = raw.normalized()
            return smoothedAim!!
        }
        val blended = (old * 0.68 + raw.normalized() * 0.32).normalized()
        smoothedAim = blended
        return blended
    }

    private fun detectTable(work: Mat): TableInfo {
        val rgb = Mat()
        val hsv = Mat()
        val mask = Mat()
        Imgproc.cvtColor(work, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
        Core.inRange(hsv, Scalar(24.0, 30.0, 25.0), Scalar(122.0, 255.0, 255.0), mask)

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(13.0, 13.0))
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel)
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel)

        val mean = Core.mean(work, mask)
        val felt = doubleArrayOf(mean.`val`[0], mean.`val`[1], mean.`val`[2], 255.0)

        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        val contourMask = mask.clone()
        Imgproc.findContours(contourMask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        val minArea = work.cols().toDouble() * work.rows().toDouble() * 0.12
        val best = contours.filter { Imgproc.contourArea(it) > minArea }.maxByOrNull { Imgproc.contourArea(it) }

        var rect = if (best != null) {
            Imgproc.boundingRect(best)
        } else if (work.cols() > work.rows()) {
            Rect(
                (work.cols() * 0.16).toInt(),
                (work.rows() * 0.16).toInt(),
                (work.cols() * 0.68).toInt(),
                (work.rows() * 0.80).toInt()
            )
        } else {
            Rect(
                (work.cols() * 0.07).toInt(),
                (work.rows() * 0.18).toInt(),
                (work.cols() * 0.86).toInt(),
                (work.rows() * 0.64).toInt()
            )
        }

        val padX = max(2, (rect.width * 0.010).toInt())
        val padY = max(2, (rect.height * 0.012).toInt())
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
        return TableInfo(rect, felt, best != null)
    }

    private fun distancePointToLine(p: Vec2, a: Vec2, b: Vec2): Double {
        val ab = b - a
        val len2 = ab.dot(ab)
        if (len2 < 1e-9) return (p - a).length()
        val t = ((p - a).dot(ab) / len2).coerceIn(0.0, 1.0)
        val q = a + ab * t
        return (p - q).length()
    }

    private fun colorDistance(color: DoubleArray, felt: DoubleArray): Double {
        if (color.size < 3 || felt.size < 3) return 0.0
        val dr = color[0] - felt[0]
        val dg = color[1] - felt[1]
        val db = color[2] - felt[2]
        return sqrt(dr * dr + dg * dg + db * db)
    }
}
