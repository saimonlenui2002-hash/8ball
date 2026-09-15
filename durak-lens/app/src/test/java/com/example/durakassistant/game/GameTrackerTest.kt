package com.example.durakassistant.game

import org.junit.Assert.*
import org.junit.Test

class GameTrackerTest {
    private val own=Deck24.cards.take(6).toSet()
    private val other=Deck24.cards.drop(6)
    private fun observation(hand:Set<Card> = own, table:List<TablePair> = emptyList(),deck:Int? = 12,
        complete:Boolean = true)=Observation(hand,table,Suit.CLUBS,deck,null,TurnPhase.UNKNOWN,.9f,
            handComplete=complete,tableComplete=true)
    private fun start():GameTracker=GameTracker().also{t->repeat(3){t.accept(observation())}}
    @Test fun coveredRoundGoesToDiscardOnce(){
        val t=start();val pair=TablePair(other[0],other[1])
        t.accept(observation(table=listOf(pair)))
        repeat(5){t.accept(observation())}
        assertEquals(setOf(other[0],other[1]),t.current().discarded)
        assertTrue(t.current().knownOpponent.isEmpty())
    }
    @Test fun disappearingUncoveredTableRequiresEvidence(){
        val t=start();t.accept(observation(table=listOf(TablePair(other[0]))))
        repeat(8){t.accept(observation())}
        assertEquals(setOf(other[0]),t.current().pendingCards)
        assertFalse(other[0] in t.current().knownOpponent)
        t.resolve(GameTracker.Outcome.OPPONENT)
        assertTrue(other[0] in t.current().knownOpponent)
        assertTrue(t.current().pendingCards.isEmpty())
    }
    @Test fun playerTakeIsNeverAnOpponentTake(){
        val t=start();t.accept(observation(table=listOf(TablePair(other[0]))))
        repeat(5){t.accept(observation(hand=own+other[0]))}
        assertTrue(t.current().knownOpponent.isEmpty())
        assertTrue(t.current().pendingCards.isEmpty())
        assertTrue(t.current().discarded.isEmpty())
    }
    @Test fun unreadDeckAndHandNeverRevealAllUnknownCards(){
        val t=start();repeat(8){t.accept(observation(hand=emptySet(),deck=null,complete=false))}
        assertEquals(12,t.current().deckCount)
        assertFalse(t.current().exactOpponent)
        assertTrue(t.current().knownOpponent.isEmpty())
    }
    @Test fun startingInMiddleCannotClaimExactHand(){
        val t=GameTracker();t.accept(observation(deck=0))
        assertFalse(t.current().exactOpponent)
    }
    @Test fun completeHistoryRevealsRemainderAfterDeckEmpty(){
        val t=start()
        for(i in 0 until 12 step 2){
            t.accept(observation(table=listOf(TablePair(other[i],other[i+1]))))
            repeat(3){t.accept(observation())}
        }
        val s=t.accept(observation(deck=0))
        assertTrue(s.exactOpponent)
        assertEquals(other.drop(12).toSet(),s.knownOpponent)
        assertEquals(12,s.discarded.size)
    }
    @Test fun pendingTakeBlocksExactHand(){
        val t=start();t.accept(observation(table=listOf(TablePair(other[0]))))
        repeat(3){t.accept(observation(deck=0))}
        assertFalse(t.current().exactOpponent)
    }
}
