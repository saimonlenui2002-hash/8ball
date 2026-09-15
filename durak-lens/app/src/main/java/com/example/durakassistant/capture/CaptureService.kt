package com.example.durakassistant.capture

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Looper
import android.view.WindowManager
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import com.example.durakassistant.MainActivity
import com.example.durakassistant.R
import com.example.durakassistant.game.GameTracker
import com.example.durakassistant.game.MoveAdvisor
import com.example.durakassistant.overlay.OverlayController
import com.example.durakassistant.vision.FrameAnalyzer

class CaptureService : Service() {
    private lateinit var workerThread: HandlerThread
    private lateinit var worker: Handler
    private val main = Handler(Looper.getMainLooper())
    private lateinit var analyzer: FrameAnalyzer
    private lateinit var overlay: OverlayController
    private val tracker = GameTracker()
    private val advisor = MoveAdvisor()
    private var projection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null
    private var lastFrameAt = 0L

    override fun onCreate() {
        super.onCreate()
        workerThread = HandlerThread("durak-frame-analysis").apply { start() }
        worker = Handler(workerThread.looper)
        analyzer = FrameAnalyzer(this)
        overlay = OverlayController(this)
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> stopCapture()
            ACTION_START -> startCapture(intent)
        }
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        releaseProjection()
        overlay.remove()
        workerThread.quitSafely()
        super.onDestroy()
    }

    private fun startCapture(intent: Intent) {
        val notification = notification()
        val serviceType = if (Build.VERSION.SDK_INT >= 29) ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION else 0
        ServiceCompat.startForeground(this, NOTIFICATION_ID, notification, serviceType)
        main.post { overlay.show() }

        val resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, 0)
        val resultData = if (Build.VERSION.SDK_INT >= 33) {
            intent.getParcelableExtra(EXTRA_RESULT_DATA, Intent::class.java)
        } else {
            @Suppress("DEPRECATION") intent.getParcelableExtra(EXTRA_RESULT_DATA)
        } ?: return stopCapture()

        releaseProjection()
        tracker.reset()
        val manager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        projection = manager.getMediaProjection(resultCode, resultData).also { mediaProjection ->
            mediaProjection.registerCallback(object : MediaProjection.Callback() {
                override fun onStop() {
                    if (projection === mediaProjection) main.post { stopCapture() }
                }
            }, main)
        }

        val windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        val bounds = windowManager.maximumWindowMetrics.bounds
        val width = bounds.width()
        val height = bounds.height()
        val density = resources.configuration.densityDpi
        imageReader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2).also { reader ->
            reader.setOnImageAvailableListener({ source -> onImage(source) }, worker)
        }
        virtualDisplay = projection?.createVirtualDisplay(
            "DurakLensCapture",
            width,
            height,
            density,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            imageReader!!.surface,
            null,
            worker
        )
    }

    private fun onImage(source: ImageReader) {
        val image = source.acquireLatestImage() ?: return
        val now = System.currentTimeMillis()
        if (now - lastFrameAt < FRAME_INTERVAL_MS) {
            image.close()
            return
        }
        lastFrameAt = now
        val bitmap = image.toBitmap()
        image.close()
        try {
            val observation = analyzer.analyze(bitmap)
            val knowledge = tracker.accept(observation)
            val advice = advisor.advise(knowledge)
            main.post { overlay.update(knowledge, advice) }
        } finally {
            bitmap.recycle()
        }
    }

    private fun Image.toBitmap(): Bitmap {
        val plane = planes[0]
        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        val paddedWidth = rowStride / pixelStride
        plane.buffer.rewind()
        val padded = Bitmap.createBitmap(paddedWidth, height, Bitmap.Config.ARGB_8888)
        padded.copyPixelsFromBuffer(plane.buffer)
        if (paddedWidth == width) return padded
        val cropped = Bitmap.createBitmap(padded, 0, 0, width, height)
        padded.recycle()
        return cropped
    }

    private fun stopCapture() {
        releaseProjection()
        main.post { overlay.remove() }
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun releaseProjection() {
        imageReader?.setOnImageAvailableListener(null, null)
        virtualDisplay?.release()
        imageReader?.close()
        val oldProjection = projection
        virtualDisplay = null
        imageReader = null
        projection = null
        oldProjection?.stop()
    }

    private fun notification(): Notification {
        val open = PendingIntent.getActivity(
            this, 1, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        val stop = PendingIntent.getService(
            this, 2, Intent(this, CaptureService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_stat_cards)
            .setContentTitle(getString(R.string.app_name))
            .setContentText(getString(R.string.capture_running))
            .setContentIntent(open)
            .setOngoing(true)
            .addAction(0, "Остановить", stop)
            .build()
    }

    private fun createNotificationChannel() {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, getString(R.string.capture_channel), NotificationManager.IMPORTANCE_LOW)
        )
    }

    companion object {
        const val ACTION_START = "com.example.durakassistant.START"
        const val ACTION_STOP = "com.example.durakassistant.STOP"
        const val EXTRA_RESULT_CODE = "result_code"
        const val EXTRA_RESULT_DATA = "result_data"
        private const val CHANNEL_ID = "capture"
        private const val NOTIFICATION_ID = 41
        private const val FRAME_INTERVAL_MS = 350L
    }
}
