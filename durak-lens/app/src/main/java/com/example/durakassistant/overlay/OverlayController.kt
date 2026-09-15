package com.example.durakassistant.overlay

import android.content.Context
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.drawable.GradientDrawable
import android.provider.Settings
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.LinearLayout
import android.widget.TextView
import com.example.durakassistant.game.*

class OverlayController(private val context:Context,
    private val onOutcome:(GameTracker.Outcome)->Unit = {}, private val onReset:()->Unit = {}) {
    private val manager=context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
    private var root:LinearLayout?=null
    private var title:TextView?=null
    private var details:TextView?=null
    private var actions:LinearLayout?=null
    private var state=GameKnowledge()
    private var page=0
    private var advice=Advice("Анализ", "Ожидаю стабильный кадр")
    private val scale get()=manager.maximumWindowMetrics.bounds.width()/720f

    fun show(){
        if(root!=null||!Settings.canDrawOverlays(context))return
        val s=scale
        val panel=LinearLayout(context).apply{
            orientation=LinearLayout.VERTICAL
            setPadding((10*s).toInt(),(5*s).toInt(),(10*s).toInt(),(5*s).toInt())
            background=GradientDrawable().apply{setColor(Color.rgb(20,40,49));cornerRadius=12*s}
        }
        fun label(size:Float)=TextView(context).apply{
            setTextColor(Color.WHITE);setTextSize(TypedValue.COMPLEX_UNIT_PX,size*s)
        }
        title=label(20f);details=label(18f).apply{maxLines=4;minLines=4}
        panel.addView(title);panel.addView(details)
        val tabs=LinearLayout(context)
        listOf("Мои","Бита","Соперник","Стол","Ход","Сброс").forEachIndexed{i,t ->
            tabs.addView(label(17f).apply{text=t;gravity=Gravity.CENTER;setOnClickListener{
                if(i==5)onReset() else {page=i;render()}
            }},LinearLayout.LayoutParams(0,(32*s).toInt(),1f))
        }
        panel.addView(tabs)
        actions=LinearLayout(context).apply{
            listOf("В биту" to GameTracker.Outcome.DISCARD,"Я взял" to GameTracker.Outcome.PLAYER,
                "Соперник взял" to GameTracker.Outcome.OPPONENT).forEach{(t,outcome)->
                addView(label(18f).apply{text=t;gravity=Gravity.CENTER;setTextColor(Color.YELLOW)
                    setOnClickListener{onOutcome(outcome)}},LinearLayout.LayoutParams(0,(34*s).toInt(),1f))
            }
        }
        panel.addView(actions)
        val layout=WindowManager.LayoutParams((700*s).toInt(),WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT).apply{gravity=Gravity.TOP or Gravity.START;x=(10*s).toInt();y=(805*s).toInt()}
        manager.addView(panel,layout);root=panel;render()
    }
    fun update(knowledge:GameKnowledge,advice:Advice){state=knowledge;this.advice=advice;if(root==null)show();render()}
    private fun render(){
        val deck=if(state.deckConfirmed)state.deckCount.toString() else "?"
        title?.text="Колода: $deck · ${if(state.exactOpponent)"Рука соперника вычислена" else "Учёт партии"}"
        val warning=state.trackingWarning.ifBlank{state.lastEvent}
        details?.text=when(page){
            0 -> "Мои (${state.hand.size}): ${cards(state.hand)}\n$warning"
            1 -> "Бита (${state.discarded.size}): ${cards(state.discarded)}\n$warning"
            2 -> if(state.exactOpponent)"У соперника: ${cards(state.knownOpponent)}\nРасчёт по полностью учтённой партии"
                else "Подтверждены: ${cards(state.knownOpponent)}\nВозможны: ${cards(state.possibleOpponent-state.knownOpponent)}\n$warning"
            4 -> if(state.trackingWarning.isEmpty())"${advice.title}\n${advice.detail}" else "Подсказка приостановлена\n$warning"
            else -> "Стол: ${cards(state.table.flatMap{listOfNotNull(it.attack,it.defense)}.toSet())}\nНе определено: ${cards(state.pendingCards)}\n$warning"
        }
        actions?.visibility=if(state.pendingCards.isEmpty())View.GONE else View.VISIBLE
    }
    private fun cards(cards:Set<Card>)=cards.sortedWith(compareBy({it.suit.ordinal},{it.rank.strength})).joinToString(" ").ifBlank{"—"}
    fun remove(){root?.let{runCatching{manager.removeView(it)}};root=null}
}
