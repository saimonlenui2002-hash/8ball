package com.example.pooltrajectory

import android.content.Context
import android.util.Base64
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import java.io.ByteArrayInputStream
import java.io.File
import java.util.zip.GZIPInputStream
import kotlin.math.max
import kotlin.math.min

/**
 * Tiny 128x128 semantic-segmentation model used only around a candidate
 * native contact marker. Classes: 0 background, 1 native guide, 2 ring.
 */
class LocalGuideDnn(private val context: Context) {
    data class Prediction(
        val labels: ByteArray,
        val intendedX: Double,
        val intendedY: Double,
        val side: Double
    ) {
        fun label(x: Int, y: Int): Int {
            if (x !in 0 until INPUT || y !in 0 until INPUT) return 0
            return labels[y * INPUT + x].toInt()
        }

        fun tablePoint(x: Double, y: Double): Vec2 = Vec2(
            intendedX + (x + 0.5) * side / INPUT.toDouble(),
            intendedY + (y + 0.5) * side / INPUT.toDouble()
        )
    }

    @Volatile private var net: Net? = null
    @Volatile var lastError: String? = null
        private set

    @Synchronized
    private fun ensureNet(): Net? {
        net?.let { return it }
        return try {
            val file = materializeModel()
            Dnn.readNetFromONNX(file.absolutePath).also {
                net = it
                lastError = null
            }
        } catch (t: Throwable) {
            lastError = t.javaClass.simpleName + ": " + (t.message ?: "model load failed")
            null
        }
    }

    private fun materializeModel(): File {
        val out = File(context.cacheDir, MODEL_FILE)
        if (out.exists() && out.length() == MODEL_BYTES) return out

        val encoded = buildString {
            for (i in 0 until MODEL_PARTS) {
                val name = MODEL_PREFIX + i.toString().padStart(2, '0')
                context.assets.open(name).bufferedReader().use { append(it.readText()) }
            }
        }
        val gz = Base64.decode(encoded, Base64.DEFAULT)
        GZIPInputStream(ByteArrayInputStream(gz)).use { input ->
            out.outputStream().use { output -> input.copyTo(output) }
        }
        if (out.length() != MODEL_BYTES) {
            throw IllegalStateException("Unexpected ONNX size ${out.length()}")
        }
        return out
    }

    @Synchronized
    fun predict(table: Mat, center: Vec2, requestedSide: Int): Prediction? {
        val n = ensureNet() ?: return null
        if (table.empty()) return null

        val side = requestedSide.coerceIn(80, 190)
        val intendedX = center.x - side * 0.5
        val intendedY = center.y - side * 0.5
        val srcX1 = max(0, intendedX.toInt())
        val srcY1 = max(0, intendedY.toInt())
        val srcX2 = min(table.cols(), kotlin.math.ceil(intendedX + side).toInt())
        val srcY2 = min(table.rows(), kotlin.math.ceil(intendedY + side).toInt())
        if (srcX2 <= srcX1 || srcY2 <= srcY1) return null

        val square = Mat.zeros(side, side, table.type())
        val src = table.submat(Rect(srcX1, srcY1, srcX2 - srcX1, srcY2 - srcY1))
        val dstX = (srcX1 - intendedX).toInt().coerceIn(0, side - 1)
        val dstY = (srcY1 - intendedY).toInt().coerceIn(0, side - 1)
        val copyW = min(src.cols(), side - dstX)
        val copyH = min(src.rows(), side - dstY)
        if (copyW <= 0 || copyH <= 0) {
            src.release(); square.release(); return null
        }
        val srcCopy = if (copyW == src.cols() && copyH == src.rows()) src else src.submat(Rect(0, 0, copyW, copyH))
        val dst = square.submat(Rect(dstX, dstY, copyW, copyH))
        srcCopy.copyTo(dst)
        dst.release()
        if (srcCopy !== src) srcCopy.release()
        src.release()

        val rgb = Mat()
        Imgproc.cvtColor(square, rgb, Imgproc.COLOR_RGBA2RGB)
        square.release()
        val blob = Dnn.blobFromImage(
            rgb,
            1.0 / 255.0,
            Size(INPUT.toDouble(), INPUT.toDouble()),
            Scalar(0.0, 0.0, 0.0),
            false,
            false
        )
        rgb.release()

        return try {
            n.setInput(blob)
            val output = n.forward()
            val flat = output.reshape(1, CLASSES)
            val rows = Array(CLASSES) { FloatArray(INPUT * INPUT) }
            for (c in 0 until CLASSES) flat.get(c, 0, rows[c])
            val labels = ByteArray(INPUT * INPUT)
            for (i in labels.indices) {
                var best = 0
                var bestValue = rows[0][i]
                for (c in 1 until CLASSES) {
                    if (rows[c][i] > bestValue) {
                        bestValue = rows[c][i]
                        best = c
                    }
                }
                labels[i] = best.toByte()
            }
            flat.release()
            output.release()
            lastError = null
            Prediction(labels, intendedX, intendedY, side.toDouble())
        } catch (t: Throwable) {
            lastError = t.javaClass.simpleName + ": " + (t.message ?: "inference failed")
            null
        } finally {
            blob.release()
        }
    }

    companion object {
        const val INPUT = 128
        private const val CLASSES = 3
        private const val MODEL_PARTS = 10
        private const val MODEL_BYTES = 73181L
        private const val MODEL_FILE = "native_contact_local_01.onnx"
        private const val MODEL_PREFIX = "native_contact_local.onnx.gz.b64."
    }
}
