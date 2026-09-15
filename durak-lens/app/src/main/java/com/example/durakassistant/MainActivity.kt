package com.example.durakassistant

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.example.durakassistant.capture.CaptureService

class MainActivity : Activity() {
    private lateinit var status: TextView
    private lateinit var overlayButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(buildUi())
        requestNotificationPermission()
    }

    override fun onResume() {
        super.onResume()
        updatePermissionState()
    }

    @Deprecated("Legacy callback is intentionally used to keep the project dependency-light")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQUEST_CAPTURE) return
        if (resultCode != RESULT_OK || data == null) {
            status.text = "Захват экрана не разрешён"
            return
        }

        val serviceIntent = Intent(this, CaptureService::class.java).apply {
            action = CaptureService.ACTION_START
            putExtra(CaptureService.EXTRA_RESULT_CODE, resultCode)
            putExtra(CaptureService.EXTRA_RESULT_DATA, data)
        }
        ContextCompat.startForegroundService(this, serviceIntent)
        status.text = "Анализ запущен. Откройте игру."
        moveTaskToBack(true)
    }

    private fun buildUi(): LinearLayout {
        val padding = (24 * resources.displayMetrics.density).toInt()
        fun text(value: String, size: Float, color: Int = Color.DKGRAY) = TextView(this).apply {
            this.text = value
            textSize = size
            setTextColor(color)
        }

        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(padding, padding, padding, padding)
            setBackgroundColor(Color.rgb(244, 248, 249))

            addView(text("Durak Lens", 30f, Color.rgb(23, 59, 74)))
            addView(text("Профиль: 24 карты · переводной · классическая колода", 16f).withMargins(0, 12, 0, 24))

            status = text("Подготовка…", 16f).also {
                it.gravity = Gravity.CENTER
                addView(it.withMargins(0, 0, 0, 20))
            }

            overlayButton = Button(context).apply {
                text = "Разрешить окно поверх игры"
                setOnClickListener { openOverlaySettings() }
            }
            addView(overlayButton, matchWidth())

            addView(Button(context).apply {
                text = "Начать анализ"
                setOnClickListener { startCaptureConsent() }
            }.withMargins(0, 12, 0, 0), matchWidth())

            addView(Button(context).apply {
                text = "Остановить"
                setOnClickListener {
                    startService(Intent(context, CaptureService::class.java).setAction(CaptureService.ACTION_STOP))
                    status.text = "Анализ остановлен"
                }
            }.withMargins(0, 8, 0, 0), matchWidth())

            addView(text(
                "При системном запросе выберите «Одно приложение», затем игру. " +
                    "Подсказка показывает только выводы из видимых карт; скрытые карты отмечаются как возможные.",
                14f
            ).withMargins(0, 24, 0, 0))
        }
    }

    private fun startCaptureConsent() {
        if (!Settings.canDrawOverlays(this)) {
            status.text = "Сначала разрешите плавающее окно"
            openOverlaySettings()
            return
        }
        val manager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        startActivityForResult(manager.createScreenCaptureIntent(), REQUEST_CAPTURE)
    }

    private fun openOverlaySettings() {
        startActivity(Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:$packageName")))
    }

    private fun updatePermissionState() {
        val allowed = Settings.canDrawOverlays(this)
        overlayButton.text = if (allowed) "Окно поверх игры: разрешено" else "Разрешить окно поверх игры"
        status.text = if (allowed) "Готово к запуску" else "Нужно разрешение на плавающее окно"
    }

    private fun requestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) !=
            android.content.pm.PackageManager.PERMISSION_GRANTED
        ) requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), REQUEST_NOTIFICATIONS)
    }

    private fun matchWidth() = LinearLayout.LayoutParams(
        ViewGroup.LayoutParams.MATCH_PARENT,
        ViewGroup.LayoutParams.WRAP_CONTENT
    )

    private fun <T : android.view.View> T.withMargins(l: Int, t: Int, r: Int, b: Int): T {
        layoutParams = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply {
            val d = resources.displayMetrics.density
            setMargins((l * d).toInt(), (t * d).toInt(), (r * d).toInt(), (b * d).toInt())
        }
        return this
    }

    companion object {
        private const val REQUEST_CAPTURE = 1001
        private const val REQUEST_NOTIFICATIONS = 1002
    }
}
