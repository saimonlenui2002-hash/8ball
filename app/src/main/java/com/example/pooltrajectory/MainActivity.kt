package com.example.pooltrajectory

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import org.opencv.android.OpenCVLoader

class MainActivity : Activity() {
    private lateinit var status: TextView
    private val prefs by lazy { getSharedPreferences(PREFS, Context.MODE_PRIVATE) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()
        if (!OpenCVLoader.initLocal()) status.text = "OpenCV не загрузился." else refreshStatus()
    }

    override fun onResume() {
        super.onResume()
        if (::status.isInitialized) refreshStatus()
    }

    private fun buildUi() {
        val scroll = ScrollView(this)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(24), dp(20), dp(32))
        }
        scroll.addView(root)

        root.addView(TextView(this).apply {
            text = "Pool Trajectory Offline 6.0 Geometry"
            textSize = 25f
            setTextColor(Color.BLACK)
        })
        root.addView(TextView(this).apply {
            text = "6.0 больше не доверяет двум маленьким выходящим веткам как источнику физики. Сначала находятся штатная входящая линия, центр битка и реальный шар на её пути. Точка контакта и обе траектории рассчитываются из геометрии шаров; короткие белые ветки используются только как дополнительная точная коррекция направления. Если геометрия не подтверждена, длинная линия специально не рисуется."
            textSize = 15f
            setPadding(0, dp(8), 0, dp(18))
        })

        status = TextView(this).apply {
            textSize = 15f
            setPadding(dp(12), dp(12), dp(12), dp(12))
            setBackgroundColor(Color.rgb(238, 238, 238))
        }
        root.addView(status, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(16) })

        root.addView(button("1. Разрешить показ поверх приложений") { requestOverlayPermission() })
        root.addView(button("2. Запустить ML 6.0") { requestCapture() })
        root.addView(button("Остановить помощник") {
            stopService(Intent(this, CaptureService::class.java))
            Toast.makeText(this, "Помощник остановлен", Toast.LENGTH_SHORT).show()
        })
        root.addView(button("Открыть 8 Ball Pool") { openGame() })

        root.addView(section("Частота анализа"))
        val fpsValue = TextView(this)
        root.addView(fpsValue)
        val fps = SeekBar(this).apply {
            max = 9
            progress = prefs.getInt(KEY_FPS, 7).coerceIn(3, 12) - 3
        }
        fun updateFps() { fpsValue.text = "${fps.progress + 3} кадров/с" }
        updateFps()
        fps.setOnSeekBarChangeListener(simpleSeek { p ->
            prefs.edit().putInt(KEY_FPS, p + 3).apply()
            updateFps()
        })
        root.addView(fps)

        root.addView(section("Рикошеты рассчитанных линий"))
        val bounceValue = TextView(this)
        root.addView(bounceValue)
        val bounces = SeekBar(this).apply {
            max = 2
            progress = prefs.getInt(KEY_BOUNCES, 0).coerceIn(0, 2)
        }
        fun updateBounces() { bounceValue.text = "${bounces.progress} отскок(а)" }
        updateBounces()
        bounces.setOnSeekBarChangeListener(simpleSeek { p ->
            prefs.edit().putInt(KEY_BOUNCES, p).apply()
            updateBounces()
        })
        root.addView(bounces)

        root.addView(CheckBox(this).apply {
            text = "Диагностика на экране: центры битка/цели, ghost-центр и входящий луч"
            isChecked = prefs.getBoolean(KEY_DEBUG, false)
            setOnCheckedChangeListener { _, checked ->
                prefs.edit().putBoolean(KEY_DEBUG, checked).apply()
            }
        }, LinearLayout.LayoutParams(-1, -2).apply { topMargin = dp(12) })

        root.addView(section("Диагностика для точной калибровки"))
        root.addView(TextView(this).apply {
            text = "Во время работы 6.0 автоматически записывает только геометрию распознавания: время кадра, центры/радиусы, направление, ghost-центр, линии, confidence и причину отказа. Скриншоты и содержимое игры в файл не сохраняются."
            textSize = 14f
            setPadding(0, 0, 0, dp(8))
        })
        root.addView(button("Экспорт диагностики в Download") {
            val name = DiagnosticRecorder.exportToDownloads(this)
            if (name != null) {
                Toast.makeText(this, "Сохранено в Download: $name", Toast.LENGTH_LONG).show()
            } else {
                Toast.makeText(this, "Диагностика пока пустая. Сначала запусти помощник и игру.", Toast.LENGTH_LONG).show()
            }
        })
        root.addView(button("Очистить диагностику") {
            DiagnosticRecorder.clear(this)
            Toast.makeText(this, "Диагностика очищена", Toast.LENGTH_SHORT).show()
        })

        root.addView(TextView(this).apply {
            text = "Для первого теста 6.0: FPS 7, рикошеты 0, экранную диагностику включить. Медленно проведи прицелом через несколько разных шаров 1–2 минуты. После теста вернись сюда, нажми «Экспорт диагностики в Download» и пришли файл вместе с записью экрана."
            textSize = 14f
            setPadding(0, dp(20), 0, 0)
        })
        setContentView(scroll)
    }

    private fun requestOverlayPermission() {
        if (Settings.canDrawOverlays(this)) {
            Toast.makeText(this, "Разрешение уже выдано", Toast.LENGTH_SHORT).show()
            return
        }
        startActivityForResult(
            Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:$packageName")),
            RC_OVERLAY
        )
    }

    private fun requestCapture() {
        if (!Settings.canDrawOverlays(this)) {
            Toast.makeText(this, "Сначала разреши показ поверх других приложений", Toast.LENGTH_LONG).show()
            requestOverlayPermission()
            return
        }
        val manager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        startActivityForResult(manager.createScreenCaptureIntent(), RC_CAPTURE)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == RC_CAPTURE && resultCode == RESULT_OK && data != null) {
            val serviceIntent = Intent(this, CaptureService::class.java).apply {
                putExtra(CaptureService.EXTRA_RESULT_CODE, resultCode)
                putExtra(CaptureService.EXTRA_PROJECTION_DATA, data)
            }
            startForegroundService(serviceIntent)
            Toast.makeText(this, "ML 6.0 запущен", Toast.LENGTH_LONG).show()
        }
        refreshStatus()
    }

    private fun openGame() {
        val candidates = listOf("com.miniclip.eightballpool", "com.miniclip.eightballpoolmult")
        val launch = candidates.firstNotNullOfOrNull { packageManager.getLaunchIntentForPackage(it) }
        if (launch != null) {
            startActivity(launch)
        } else {
            Toast.makeText(this, "8 Ball Pool не найден", Toast.LENGTH_LONG).show()
        }
    }

    private fun refreshStatus() {
        status.text = if (Settings.canDrawOverlays(this)) {
            val diag = if (DiagnosticRecorder.hasData(this)) " • диагностика накоплена" else ""
            "✓ Overlay разрешён. Версия 6.0 • geometry-first$diag."
        } else {
            "Нужно разрешение «поверх других приложений»."
        }
    }

    private fun button(text: String, action: () -> Unit): View = Button(this).apply {
        this.text = text
        isAllCaps = false
        setOnClickListener { action() }
        gravity = Gravity.CENTER
    }

    private fun section(text: String) = TextView(this).apply {
        this.text = text
        textSize = 16f
        setTextColor(Color.BLACK)
        setPadding(0, dp(18), 0, dp(4))
    }

    private fun simpleSeek(onChanged: (Int) -> Unit) = object : SeekBar.OnSeekBarChangeListener {
        override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
            if (fromUser) onChanged(progress)
        }
        override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit
        override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
    }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    companion object {
        const val PREFS = "pool_prefs"
        const val KEY_SENSITIVITY = "sensitivity"
        const val KEY_FPS = "fps"
        const val KEY_BOUNCES = "bounces"
        const val KEY_DEBUG = "debug"
        private const val RC_OVERLAY = 100
        private const val RC_CAPTURE = 101
    }
}
