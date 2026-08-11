package com.example.pooltrajectory

import android.content.ContentValues
import android.content.Context
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object DiagnosticRecorder {
    private const val INTERNAL_FILE = "pool_v6_diagnostics.tsv"
    private const val MAX_BYTES = 4L * 1024L * 1024L

    @Synchronized
    fun startSession(context: Context, width: Int, height: Int) {
        val file = File(context.filesDir, INTERNAL_FILE)
        if (!file.exists() || file.length() > MAX_BYTES) {
            file.writeText(header())
        }
        file.appendText(
            "# session\t${System.currentTimeMillis()}\tframe=${width}x${height}\tversion=6.0\n"
        )
    }

    @Synchronized
    fun append(context: Context, result: AnalysisResult) {
        val file = File(context.filesDir, INTERNAL_FILE)
        if (!file.exists()) file.writeText(header())
        if (file.length() > MAX_BYTES) {
            file.writeText(header() + "# rotated\t${System.currentTimeMillis()}\n")
        }

        val cue = result.cueBall
        val target = result.targetBall
        val ghost = result.ghostCueCenter
        val aim = result.aimDirection
        val segments = result.segments.joinToString("|") { s ->
            "${s.kind}:${f(s.start.x)},${f(s.start.y)}>${f(s.end.x)},${f(s.end.y)}"
        }
        val safeStatus = result.status.replace('\t', ' ').replace('\n', ' ')
        val safeDetails = result.diagnostics.replace('\t', ' ').replace('\n', ' ')

        file.appendText(
            buildString {
                append(System.currentTimeMillis()).append('\t')
                append(result.confidence).append('\t')
                append(safeStatus).append('\t')
                append(cue?.let { "${f(it.center.x)},${f(it.center.y)},${f(it.radius)}" } ?: "-").append('\t')
                append(target?.let { "${f(it.center.x)},${f(it.center.y)},${f(it.radius)}" } ?: "-").append('\t')
                append(ghost?.let { "${f(it.x)},${f(it.y)}" } ?: "-").append('\t')
                append(aim?.let { "${f(it.x)},${f(it.y)}" } ?: "-").append('\t')
                append(segments).append('\t')
                append(safeDetails)
                append('\n')
            }
        )
    }

    fun hasData(context: Context): Boolean {
        val file = File(context.filesDir, INTERNAL_FILE)
        return file.exists() && file.length() > header().length + 16
    }

    @Synchronized
    fun clear(context: Context) {
        File(context.filesDir, INTERNAL_FILE).writeText(header())
    }

    @Synchronized
    fun exportToDownloads(context: Context): String? {
        val source = File(context.filesDir, INTERNAL_FILE)
        if (!source.exists() || source.length() <= header().length + 16) return null

        val stamp = SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", Locale.US).format(Date())
        val name = "PoolTrajectory-Diagnostic-$stamp.tsv"
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "text/tab-separated-values")
            put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
        }
        val uri = context.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            ?: return null
        return try {
            context.contentResolver.openOutputStream(uri)?.use { out ->
                source.inputStream().use { input -> input.copyTo(out) }
            } ?: return null
            name
        } catch (_: Throwable) {
            runCatching { context.contentResolver.delete(uri, null, null) }
            null
        }
    }

    private fun header() =
        "timestamp_ms\tconfidence\tstatus\tcue_xyr\ttarget_xyr\tghost_xy\taim_xy\tsegments\tdetails\n"

    private fun f(v: Double) = String.format(Locale.US, "%.3f", v)
}
