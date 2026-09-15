package com.example.durakassistant.game

class GameTracker {
    private var state = GameKnowledge()

    @Synchronized
    fun reset() {
        state = GameKnowledge()
    }

    @Synchronized
    fun current(): GameKnowledge = state

    @Synchronized
    fun accept(observation: Observation): GameKnowledge {
        val previous = state
        val previousTable = previous.table.flatMap { listOfNotNull(it.attack, it.defense) }.toSet()
        val currentTable = observation.tableCards
        val observedDeck = observation.deckCount ?: previous.deckCount
        var discarded = previous.discarded
        var knownOpponent = previous.knownOpponent

        // A cleared table either went to discard or was taken. A rise in the
        // opponent's hand count is the reliable signal that the opponent took it.
        if (previousTable.isNotEmpty() && currentTable.isEmpty()) {
            val observedCount = observation.opponentCount
            val playerTook = observation.hand.any { it in previousTable }
            val deckWasRefilled = observedDeck < previous.deckCount
            if (!playerTook && !deckWasRefilled && previous.deckCount > 0) {
                knownOpponent = knownOpponent + previousTable
            } else if (deckWasRefilled || (observedCount != null && observedCount <= previous.opponentCount)) {
                discarded = discarded + previousTable
            }
        }

        // Once an already known card appears on the table, it has left the hand.
        knownOpponent = knownOpponent - currentTable
        discarded = discarded - currentTable

        val hand = stableSet(observation.hand, previous.hand, observation.confidence)
        val trump = observation.trump ?: previous.trump
        val deckCount = observedDeck
        val opponentCount = observation.opponentCount ?: (
            Deck24.cards.size - deckCount - hand.size - discarded.size - currentTable.size
        ).coerceIn(0, Deck24.cards.size)

        val impossible = hand + currentTable + discarded
        val candidates = Deck24.cards - impossible
        knownOpponent = knownOpponent.intersect(candidates)

        // With an empty deck, if the candidate count equals the number of hidden
        // cards, elimination has determined the complete opponent hand.
        if (deckCount == 0 && candidates.size == opponentCount) knownOpponent = candidates

        state = GameKnowledge(
            hand = hand,
            table = observation.table,
            trump = trump,
            deckCount = deckCount.coerceIn(0, 12),
            opponentCount = opponentCount.coerceAtLeast(0),
            knownOpponent = knownOpponent,
            possibleOpponent = candidates,
            discarded = discarded,
            phase = resolvePhase(observation),
            confidence = observation.confidence
        )
        return state
    }

    private fun stableSet(new: Set<Card>, old: Set<Card>, confidence: Float): Set<Card> = when {
        confidence >= 0.72f -> new
        new.isEmpty() -> old
        else -> new + old.intersect(new)
    }

    private fun resolvePhase(observation: Observation): TurnPhase {
        if (observation.phase != TurnPhase.UNKNOWN) return observation.phase
        val attacks = observation.table.size
        val defenses = observation.table.count { it.defense != null }
        return when {
            attacks == 0 -> TurnPhase.ATTACK
            attacks > defenses -> TurnPhase.DEFEND
            attacks == defenses -> TurnPhase.ATTACK
            else -> TurnPhase.WAIT
        }
    }
}
