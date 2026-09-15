package com.example.durakassistant.game

/** Tracks a whole round; a single empty frame is not an opponent take. */
class GameTracker {
    enum class Outcome { DISCARD, PLAYER, OPPONENT }
    private var state=GameKnowledge()
    private var roundCards=emptySet<Card>()
    private var coveredCards=emptySet<Card>()
    private var emptyFrames=0
    private var started=false
    private var historyComplete=false
    private var startVotes=0

    @Synchronized fun reset(){
        state=GameKnowledge();roundCards=emptySet();coveredCards=emptySet()
        emptyFrames=0;started=false;historyComplete=false;startVotes=0
    }
    @Synchronized fun current():GameKnowledge=state

    @Synchronized fun resolve(outcome:Outcome):GameKnowledge {
        val cards=state.pendingCards
        if(cards.isEmpty())return state
        state=when(outcome){
            Outcome.DISCARD -> state.copy(discarded=state.discarded+cards,
                knownOpponent=state.knownOpponent-cards,lastEvent="Подтверждена бита")
            Outcome.PLAYER -> state.copy(hand=state.hand+cards,
                knownOpponent=state.knownOpponent-cards,lastEvent="Вы взяли стол")
            Outcome.OPPONENT -> state.copy(knownOpponent=state.knownOpponent+cards,
                opponentTaken=state.opponentTaken+cards,lastEvent="Соперник взял стол")
        }.copy(pendingCards=emptySet(),exactOpponent=false)
        return state
    }

    @Synchronized fun accept(o:Observation):GameKnowledge {
        val newDeal=o.deckCount==12 && o.hand.size==6 && o.handComplete && o.tableCards.isEmpty()
        if(newDeal && (!started || state.deckCount<=2)){
            startVotes++
            if(startVotes>=3){
                reset();started=true;historyComplete=true
                state=GameKnowledge(hand=o.hand,trump=o.trump,deckCount=12,deckConfirmed=true,
                    lastEvent="Новая партия: 24 карты",trackingWarning="")
            }
        }else startVotes=0

        // Retain confirmed zones across occlusion/animations. An unread hand
        // means incomplete observation, not that every missing card is hidden.
        if(!o.handComplete){
            state=state.copy(exactOpponent=false,phase=TurnPhase.UNKNOWN,
                trackingWarning="Рука не полностью видна — жду стабильный кадр")
            return state
        }
        val previous=state
        var discarded=previous.discarded
        var known=previous.knownOpponent
        var taken=previous.opponentTaken
        var pending=previous.pendingCards
        var event=previous.lastEvent
        val conflicts=discarded.intersect(o.hand+o.tableCards)
        if(conflicts.isNotEmpty()){
            discarded-=conflicts;historyComplete=false
            event="Есть противоречие с битой — точный расчёт остановлен"
        }
        val deck=o.deckCount?:previous.deckCount
        if(previous.deckConfirmed && o.deckCount!=null && deck>previous.deckCount && !newDeal){
            state=previous.copy(exactOpponent=false,trackingWarning="Проверяю счётчик колоды")
            return state
        }
        if(o.tableCards.isNotEmpty()){
            emptyFrames=0
            if(pending.isNotEmpty()){
                // A table reappearing immediately is an occlusion, not a take.
                if(o.tableCards.any{it in pending}){roundCards+=pending;pending=emptySet()}
                else {historyComplete=false;event="Предыдущий стол требует подтверждения"}
            }
            roundCards+=o.tableCards
            for(pair in o.table) if(pair.defense!=null)coveredCards+=setOf(pair.attack,pair.defense)
            known-=o.tableCards
        }else if(o.tableComplete){
            emptyFrames++
            if(emptyFrames>=3 && roundCards.isNotEmpty()){
                val playerReceived=roundCards.all{it in o.hand}
                when {
                    playerReceived -> event="Вы взяли: ${format(roundCards)}"
                    coveredCards.containsAll(roundCards) -> {
                        discarded+=roundCards;known-=roundCards
                        event="Бита: ${format(roundCards)}"
                    }
                    else -> {pending+=roundCards;event="Кто забрал стол? Подтвердите ниже"}
                }
                roundCards=emptySet();coveredCards=emptySet()
            }
        }
        if(pending.isNotEmpty() && pending.all{it in o.hand}){
            event="Вы взяли: ${format(pending)}";pending=emptySet()
        }
        val missingFromHand=previous.hand-o.hand
        if(previous.hand.isNotEmpty() && missingFromHand.any{it !in roundCards && it !in discarded && it !in o.tableCards && it !in pending}){
            historyComplete=false
        }
        val table=o.tableCards
        val hand=o.hand-table
        known-=hand+table+discarded
        val possible=Deck24.cards-hand-table-discarded-pending
        val exact=historyComplete && pending.isEmpty() && deck==0 && o.deckCount==0 && o.tableComplete
        if(exact)known=possible
        val count=(24-deck-hand.size-table.size-discarded.size-pending.size).coerceIn(0,possible.size)
        state=GameKnowledge(hand=hand,table=o.table,trump=o.trump?:previous.trump,
            deckCount=deck,deckConfirmed=o.deckCount!=null||previous.deckConfirmed,
            opponentCount=count,knownOpponent=known.intersect(possible),possibleOpponent=possible,
            discarded=discarded,opponentTaken=taken,pendingCards=pending,exactOpponent=exact,
            lastEvent=event,phase=if(pending.isEmpty())o.phase else TurnPhase.UNKNOWN,
            confidence=o.confidence,trackingWarning=when{
                pending.isNotEmpty()->"Не определён получатель стола"
                !historyComplete->"Учёт неполный: точная рука не подтверждена"
                o.deckCount==null->"Счётчик сейчас не читается"
                !o.tableComplete->"Не все карты стола распознаны"
                else->""
            })
        return state
    }
    private fun format(cards:Set<Card>)=cards.sortedWith(compareBy({it.suit.ordinal},{it.rank.strength})).joinToString(" ")
}
