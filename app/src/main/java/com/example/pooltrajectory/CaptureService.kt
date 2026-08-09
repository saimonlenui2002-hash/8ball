package com.example.pooltrajectory

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
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Looper
import android.provider.Settings
import org.opencv.android.OpenCVLoader
import java.util.concurrent.atomic.AtomicBoolean

class CaptureService : Service() {
    private var projection: MediaProjection? = null
    private var display: VirtualDisplay? = null
    private var reader: ImageReader? = null
    private var thread: HandlerThread? = null
    private var handler: Handler? = null
    private val main = Handler(Looper.getMainLooper())
    private val busy = AtomicBoolean(false)
    private var overlay: OverlayController? = null
    private var analyzer: NativeGuideForkAnalyzerV32? = null
    private var last = 0L
    private var captureWidth = 0
    private var captureHeight = 0
    private var densityDpi = 0

    override fun onCreate() {
        super.onCreate()
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(CHANNEL, "Захват экрана", NotificationManager.IMPORTANCE_LOW)
        )
        OpenCVLoader.initLocal()
        analyzer = NativeGuideForkAnalyzerV32(this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopSelf()
            return START_NOT_STICKY
        }
        val code = intent?.getIntExtra(EXTRA_RESULT_CODE, Int.MIN_VALUE) ?: Int.MIN_VALUE
        @Suppress("DEPRECATION")
        val data = intent?.getParcelableExtra<Intent>(EXTRA_PROJECTION_DATA)
        if (code == Int.MIN_VALUE || data == null || !Settings.canDrawOverlays(this)) {
            stopSelf()
            return START_NOT_STICKY
        }
        startForeground(ID, notification(), ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION)
        if (projection == null) startProjection(code, data)
        return START_NOT_STICKY
    }

    private fun startProjection(code: Int, data: Intent) {
        val mgr = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        val p = mgr.getMediaProjection(code, data) ?: run {
            stopSelf()
            return
        }
        projection = p
        thread = HandlerThread("pool-capture").also { it.start() }
        handler = Handler(thread!!.looper)

        val dm = resources.displayMetrics
        densityDpi = dm.densityDpi
        captureWidth = dm.widthPixels
        captureHeight = dm.heightPixels

        p.registerCallback(object : MediaProjection.Callback() {
            override fun onStop() {
                stopSelf()
            }

            override fun onCapturedContentResize(width: Int, height: Int) {
                if (width <= 0 || height <= 0) return
                handler?.post { resizeCapture(width, height) }
            }
        }, handler)

        reader = createReader(captureWidth, captureHeight)
        display = p.createVirtualDisplay(
            "PoolTrajectoryCapture",
            captureWidth,
            captureHeight,
            densityDpi,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            reader!!.surface,
            null,
            handler
        )
        overlay = OverlayController(this).also { it.show() }
    }

    private fun resizeCapture(width: Int, height: Int) {
        if (width == captureWidth && height == captureHeight) return
        val virtualDisplay = display ?: return
        val oldReader = reader
        val newReader = createReader(width, height)
        try {
            virtualDisplay.resize(width, height, densityDpi)
            virtualDisplay.setSurface(newReader.surface)
            reader = newReader
            captureWidth = width
            captureHeight = height
            oldReader?.setOnImageAvailableListener(null, null)
            oldReader?.close()
            busy.set(false)
            last = 0L
        } catch (_: Throwable) {
            newReader.setOnImageAvailableListener(null, null)
            newReader.close()
        }
    }

    private fun createReader(width: Int, height: Int): ImageReader {
        return ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2).also { imageReader ->
            imageReader.setOnImageAvailableListener({ r -> onImage(r) }, handler)
        }
    }

    private fun onImage(r: ImageReader) {
        val image = r.acquireLatestImage() ?: return
        val fps = getSharedPreferences(MainActivity.PREFS, Context.MODE_PRIVATE)
            .getInt(MainActivity.KEY_FPS, 7)
            .coerceIn(3, 12)
        val now = System.currentTimeMillis()
        if (now - last < 1000L / fps || !busy.compareAndSet(false, true)) {
            image.close()
            return
        }
        last = now
        try {
            val bmp = imageToBitmap(image)
            val result = analyzer?.analyze(bmp)
            bmp.recycle()
            if (result != null) main.post { overlay?.update(result) }
        } catch (_: Throwable) {
        } finally {
            image.close()
            busy.set(false)
        }
    }

    private fun imageToBitmap(image: Image): Bitmap {
        val plane = image.planes[0]
        val buffer = plane.buffer
        buffer.rewind()
        val paddedWidth = image.width +
            (plane.rowStride - plane.pixelStride * image.width) / plane.pixelStride
        val padded = Bitmap.createBitmap(paddedWidth, image.height, Bitmap.Config.ARGB_8888)
        padded.copyPixelsFromBuffer(buffer)
        if (paddedWidth == image.width) return padded
        val out = Bitmap.createBitmap(padded, 0, 0, image.width, image.height)
        padded.recycle()
        return out
    }

    private fun notification(): Notification {
        val stop = PendingIntent.getService(
            this, 2,
            Intent(this, CaptureService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val open = PendingIntent.getActivity(
            this, 1,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        return Notification.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.ic_menu_view)
            .setContentTitle("Pool Trajectory Offline")
            .setContentText("Анализ контактного узла 3.2 активен")
            .setContentIntent(open)
            .setOngoing(true)
            .addAction(android.R.drawable.ic_menu_close_clear_cancel, "Остановить", stop)
            .build()
    }

    override fun onDestroy() {
        overlay?.hide()
        reader?.setOnImageAvailableListener(null, null)
        display?.setSurface(null)
        display?.release()
        reader?.close()
        runCatching { projection?.stop() }
        thread?.quitSafely()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    companion object {
        const val EXTRA_RESULT_CODE = "resultCode"
        const val EXTRA_PROJECTION_DATA = "projectionData"
        const val ACTION_STOP = "com.example.pooltrajectory.STOP"
        private const val CHANNEL = "screen_capture"
        private const val ID = 73
    }
}
