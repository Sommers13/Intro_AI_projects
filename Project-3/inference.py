# -*- coding: utf-8 -*-
# inference.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import itertools
import util
import random
import busters
import game

class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    This is an abstract class, which you should not modify.
    """

    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        "Sets the ghost agent for later access"
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = [] # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistribution(self, gameState):
        """
        Returns a distribution over successor positions of the ghost from the given gameState.

        You must first place the ghost in the gameState, using setGhostPosition below.
        """
        ghostPosition = gameState.getGhostPosition(self.index) # The position you set
        actionDist = self.ghostAgent.getDistribution(gameState)
        dist = util.Counter()
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            dist[successorPosition] = prob
        return dist

    def setGhostPosition(self, gameState, ghostPosition):
        """
        Sets the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of the
        GameState object which is responsible for maintaining game state, not a
        reference to the original object.  Note also that the ghost distance
        observations are stored at the time the GameState object is created, so
        changing the position of the ghost will not affect the functioning of
        observeState.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[self.index] = game.AgentState(conf, False)
        return gameState

    def observeState(self, gameState):
        "Collects the relevant noisy distance observation and pass it along."
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index: # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observe(obs, gameState)

    def initialize(self, gameState):
        "Initializes beliefs to a uniform distribution over all positions."
        # The legal positions do not include the ghost prison cells in the bottom left.
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        "Sets the belief state to a uniform prior belief over all positions."
        pass

    def observe(self, observation, gameState):
        "Updates beliefs based on the given distance observation and gameState."
        pass

    def elapseTime(self, gameState):
        "Updates beliefs for a time step elapsing from a gameState."
        pass

    def getBeliefDistribution(self):
        """
        Returns the agent's current belief state, a distribution over
        ghost locations conditioned on all evidence so far.
        """
        pass

class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward-algorithm
    updates to compute the exact belief function at each time step.
    """

    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        """
        Updates beliefs based on the distance observation and Pacman's position.

        The noisyDistance is the estimated manhattan distance to the ghost you are tracking.

        The emissionModel below stores the probability of the noisyDistance for any true
        distance you supply.  That is, it stores P(noisyDistance | TrueDistance).

        self.legalPositions is a list of the possible ghost positions (you
        should only consider positions that are in self.legalPositions).

        A correct implementation will handle the following special case:
          *  When a ghost is captured by Pacman, all beliefs should be updated so
             that the ghost appears in its prison cell, position self.getJailPosition()

             You can check if a ghost has been captured by Pacman by
             checking if it has a noisyDistance of None (a noisy distance
             of None will be returned if, and only if, the ghost is
             captured).

        """

        # variables given
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()

        # Replace this code with a correct observation update
        # Be sure to handle the "jail" edge case where the ghost is eaten
        # and noisyDistance is None
        allPossible = util.Counter()

        #for p in self.legalPositions:
        #    trueDistance = util.manhattanDistance(p, pacmanPosition)
        #    if emissionModel[trueDistance] > 0: allPossible[p] = 1.0

        # QUESTION 1

        if (noisyDistance is not None): # if we have an observation
            # allPossible = util.Counter()
            # for each position we can be in
            for p in self.legalPositions:
                trueDistance = util.manhattanDistance(p, pacmanPosition)
                allPossible[p] = self.beliefs[p] * emissionModel[trueDistance]
        else: # if we do not have an observation
            allPossible = util.Counter()
            # deal with the edge case for jail position
            allPossible[self.getJailPosition()] = 1.0

        # given
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        """
        Update self.beliefs in response to a time step passing from the current state.

        The transition model is not entirely stationary: it may depend on Pacman's
        current position (e.g., for DirectionalGhost).  However, this is not a problem,
        as Pacman's current position is known.

        In order to obtain the distribution over new positions for the
        ghost, given its previous position (oldPos) as well as Pacman's
        current position, use this line of code:

          newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))

        Note that you may need to replace "oldPos" with the correct name
        of the variable that you have used to refer to the previous ghost
        position for which you are computing this distribution. You will need to compute
        multiple position distributions for a single update.

        newPosDist is a util.Counter object, where for each position p in self.legalPositions,

        newPostDist[p] = Pr( ghost is at position p at time t + 1 | ghost is at position oldPos at time t )

        (and also given Pacman's current position).  You may also find it useful to loop over key, value pairs
        in newPosDist, like:

          for newPos, prob in newPosDist.items():
            ...

        *** GORY DETAIL AHEAD ***

        As an implementation detail (with which you need not concern
        yourself), the line of code at the top of this comment block for obtaining newPosDist makes
        use of two helper methods provided in InferenceModule above:

          1) self.setGhostPosition(gameState, ghostPosition)
              This method alters the gameState by placing the ghost we're tracking
              in a particular position.  This altered gameState can be used to query
              what the ghost would do in this position.

          2) self.getPositionDistribution(gameState)
              This method uses the ghost agent to determine what positions the ghost
              will move to from the provided gameState.  The ghost must be placed
              in the gameState with a call to self.setGhostPosition above.

        It is worthwhile, however, to understand why these two helper methods are used and how they
        combine to give us a belief distribution over new positions after a time update from a particular position
        """

#        util.raiseNotDefined()

        #return

        allPosibilities = util.Counter()

        # for every position
        for p in self.legalPositions:
            distribution = self.getPositionDistribution(self.setGhostPosition(gameState,p))
            for pos, prob in distribution.items():
                allPosibilities[pos] = allPosibilities[pos] + self.beliefs[p] * prob

        self.beliefs = allPosibilities

    def getBeliefDistribution(self):
        return self.beliefs

class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.

    Useful helper functions will include random.choice, which chooses
    an element from a list uniformly at random, and util.sample, which
    samples a key from a Counter by treating its values as probabilities.
    """


    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent);
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles


    def initializeUniformly(self, gameState):
        """
          Initializes a list of particles. Use self.numParticles for theasd number of particles.
          Use self.legalPositions for the legal board positions where a particle could be located.
          Particles should be evenly (not randomly) distributed across positions in order to
          ensure a uniform prior.

          ** NOTE **
            the variable you store your particles in must be a list; a list is simply a collection
            of unweighted variables (positions in this case). Storing your particles as a Counter or
            dictionary (where there could be an associated weight with each position) is incorrect
            and will produce errors
        """


        #   ---------------------------- QUESTION 4

        # initialize the particles as bubbles bc cuter that way
        self.bubbles = []
        holder = 0

        while holder < self.numParticles: # MAKE SURE WE GET ALL THE BUBBLES
            for pos in self.legalPositions:
                if holder < self.numParticles:
                    holder = holder +1
                    self.bubbles.append(pos)


    def observe(self, observation, gameState):
        """
        Update beliefs based on the given distance observation. Make
        sure to handle the special case where all particles have weight
        0 after reweighting based on observation. If this happens,
        resample particles uniformly at random from the set of legal
        positions (self.legalPositions).

        A correct implementation will handle two special cases:
          1) When a ghost is captured by Pacman, **all** particles should be updated so
             that the ghost appears in its prison cell, self.getJailPosition()

             You can check if a ghost has been captured by Pacman by
             checking if it has a noisyDistance of None (a noisy distance
             of None will be returned if, and only if, the ghost is
             captured).

          2) When all particles receive 0 weight, they should be recreated from the
             prior distribution by calling initializeUniformly. The total weight
             for a belief distribution can be found by calling totalCount on
             a Counter object

        util.sample(Counter object) is a helper method to generate a sample from
        a belief distribution

        You may also want to use util.manhattanDistance to calculate the distance
        between a particle and pacman's position.
        """

        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()

        #   ---------------------------- QUESTION 4

        if (noisyDistance is not None): # if we have an observation
            allPossible = util.Counter()
            bdistribution = self.getBeliefDistribution()
            # for each position we can be in
            for p in self.legalPositions:
                trueDistance = util.manhattanDistance(p, pacmanPosition)
                #allPossible[p] = allPossible[p] + emissionModel[trueDistance]
                allPossible[p] = allPossible[p] + emissionModel[trueDistance]*bdistribution[p]

            if not any (allPossible.values()):
                self.initializeUniformly(gameState)
                return

            # so we have values lets make them our new bubbles
            bubbles = []
            i = 0

            while i < self.numParticles:
                bubbles.append(util.sample(allPossible))
                i = i +1

            self.bubbles = bubbles

        else: # if we do not have an observation
            #allPossible = util.Counter()
            # deal with the edge case for jail position
            #allPossible[self.getJailPosition()] = 1.0

            self.bubbles = [self.getJailPosition()] * self.numParticles


    def elapseTime(self, gameState):
        """
        Update beliefs for a time step elapsing.

        As in the elapseTime method of ExactInference, you should use:

          newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))

        to obtain the distribution over new positions for the ghost, given
        its previous position (oldPos) as well as Pacman's current
        position.

        util.sample(Counter object) is a helper method to generate a sample from a
        belief distribution
        """

        #   ---------------------------- QUESTION 5
        bubbles = []

        for bubble in self.bubbles:
            newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, bubble))
            #bubbles.append(newPosDist)
            bubbles.append(util.sample(newPosDist))

        # update paramater with calculations
        self.bubbles = bubbles

    def getBeliefDistribution(self):
        """
          Return the agent's current belief state, a distribution over
          ghost locations conditioned on all evidence and time passage. This method
          essentially converts a list of particles into a belief distribution (a Counter object)
        """

        #   ---------------------------- QUESTION 4
        distribubbles = util.Counter() # belief distribution Counter object

        for i  in self.bubbles: # for each bubble
            distribubbles[i] = distribubbles[i] + 1 # increase by 1

        distribubbles.normalize() # normalize otherwise the whole thing fucks up

        # return the distribution bc getter
        return distribubbles

class MarginalInference(InferenceModule):
    "A wrapper around the JointInference module that returns marginal beliefs about ghosts."

    def initializeUniformly(self, gameState):
        "Set the belief state to an initial, prior value."
        if self.index == 1: jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observeState(self, gameState):
        "Update beliefs based on the given distance observation and gameState."
        if self.index == 1: jointInference.observeState(gameState)

    def elapseTime(self, gameState):
        "Update beliefs for a time step elapsing from a gameState."
        if self.index == 1: jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        "Returns the marginal belief over a particular ghost by summing out the others."
        jointDistribution = jointInference.getBeliefDistribution()
        dist = util.Counter()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist

class JointParticleFilter:
    "JointParticleFilter tracks a joint distribution over tuples of all ghost positions."

    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initialize(self, gameState, legalPositions):
        "Stores information about the game, then initializes particles."
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeParticles()

    def initializeParticles(self):
        """
        Initialize particles to be consistent with a uniform prior.

        Each particle is a tuple of ghost positions. Use self.numParticles for
        the number of particles. You may find the python package 'itertools' helpful.
        Specifically, you will need to think about permutations of legal ghost
        positions, with the additional understanding that ghosts may occupy the
        same space. Look at the 'product' function in itertools to get an
        implementation of the catesian product. Note: If you use
        itertools, keep in mind that permutations are not returned in a random order;
        you must shuffle the list of permutations in order to ensure even placement
        of particles across the board. Use self.legalPositions to obtain a list of
        positions a ghost may occupy.

          ** NOTE **
            the variable you store your particles in must be a list; a list is simply a collection
            of unweighted variables (positions in this case). Storing your particles as a Counter or
            dictionary (where there could be an associated weight with each position) is incorrect
            and will produce errors

        """

        # QUESTION 6 -----------------------------------------------------------
        ghosts =  list (itertools.product(self.legalPositions,repeat=self.numGhosts))
        random.shuffle(ghosts)

        # from question 4

        self.bubbles = [] # BUBBULES ARE WAY CUTER
        holder = 0

        while holder < self.numParticles:  # MAKE SURE WE GET ALL THE BUBBLES
            for pos in ghosts: # GO THROUGH THE GHOSTS DON'T CHANGE THIS STUPID ERROR
                if holder < self.numParticles:
                    holder = holder + 1
                    self.bubbles.append(pos)

    def addGhostAgent(self, agent):
        "Each ghost agent is registered separately and stored (in case they are different)."
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1);

    def observeState(self, gameState):
        """
        Resamples the set of particles using the likelihood of the noisy observations.

        To loop over the ghosts, use:

          for i in range(self.numGhosts):
            ...

        A correct implementation will handle two special cases:
          1) When a ghost is captured by Pacman, all particles should be updated so
             that the ghost appears in its prison cell, position self.getJailPosition(i)
             where "i" is the index of the ghost.

             You can check if a ghost has been captured by Pacman by
             checking if it has a noisyDistance of None (a noisy distance
             of None will be returned if, and only if, the ghost is
             captured).

          2) When all particles receive 0 weight, they should be recreated from the
              prior distribution by calling initializeParticles. After all particles
              are generated randomly, any ghosts that are eaten (have noisyDistance of 0)
              must be changed to the jail Position. This will involve changing each
              particle if a ghost has been eaten.

        ** Remember ** We store particles as tuples, but to edit a specific particle,
        it must be converted to a list, edited, and then converted back to a tuple. Since
        this is a common operation when placing a ghost in the jail for a particle, we have
        provided a helper method named self.getParticleWithGhostInJail(particle, ghostIndex)
        that performs these three operations for you.

        """
        pacmanPosition = gameState.getPacmanPosition()
        noisyDistances = gameState.getNoisyGhostDistances()
        if len(noisyDistances) < self.numGhosts: return
        emissionModels = [busters.getObservationDistribution(dist) for dist in noisyDistances]

        # QUESTION 6 -----------------------------------------------------------
            # question 6 hard part

        posibililes = util.Counter()

        # go through all bubbles ----------------------------------------------------------
        for bubble in self.bubbles:

            emissions_tmp = 1

            ghost_counter = 0 # ghost counter

            while ghost_counter < self.numGhosts:  # go through each ghost
                if noisyDistances[ghost_counter] is not None: # so we have a reading
                    #trueDistance = util.manhattanDistance(bubble, pacmanPosition)
                    trueDistance = util.manhattanDistance(bubble[ghost_counter], pacmanPosition)

                    emissions_tmp = emissions_tmp * emissionModels[ghost_counter][trueDistance]


                else: # we do not have a reading
                    bubble = self.getParticleWithGhostInJail(bubble, ghost_counter)

                # increase index
                ghost_counter = ghost_counter + 1

            # add to the possibile list
            posibililes[bubble] = posibililes[bubble] + emissions_tmp

        # end going through the self.bubbles ---------------------------------------------

        if not any(posibililes.values()):  # something works

            self.initializeParticles()

            for bubble in self.bubbles:
                ghost_counter = 0  # ghost counter

                while ghost_counter < self.numGhosts:  # go through each ghost
                    if noisyDistances[ghost_counter] is None:  # so we have a reading
                        # trueDistance = util.manhattanDistance(bubble, pacmanPosition)
                        bubble = self.getParticleWithGhostInJail(bubble, ghost_counter)

                    ghost_counter = ghost_counter +1

            return

        # so we do have stuff that works

        posibililes.normalize()

        # so we have values lets make them our new bubbles
        bubbles = []
        i = 0

        while i < self.numParticles:
            bubbles.append(util.sample(posibililes))
            i = i + 1

        self.bubbles = bubbles




    def getParticleWithGhostInJail(self, particle, ghostIndex):
        particle = list(particle)
        particle[ghostIndex] = self.getJailPosition(ghostIndex)
        return tuple(particle)

    def elapseTime(self, gameState):
        """
        Samples each particle's next state based on its current state and the gameState.

        To loop over the ghosts, use:

          for i in range(self.numGhosts):
            ...

        Then, assuming that "i" refers to the index of the
        ghost, to obtain the distributions over new positions for that
        single ghost, given the list (prevGhostPositions) of previous
        positions of ALL of the ghosts, use this line of code:

          newPosDist = getPositionDistributionForGhost(setGhostPositions(gameState, prevGhostPositions),
                                                       i, self.ghostAgents[i])

        **Note** that you may need to replace "prevGhostPositions" with the
        correct name of the variable that you have used to refer to the
        list of the previous positions of all of the ghosts, and you may
        need to replace "i" with the variable you have used to refer to
        the index of the ghost for which you are computing the new
        position distribution.

        As an implementation detail (with which you need not concern
        yourself), the line of code above for obtaining newPosDist makes
        use of two helper functions defined below in this file:

          1) setGhostPositions(gameState, ghostPositions)
              This method alters the gameState by placing the ghosts in the supplied positions.

          2) getPositionDistributionForGhost(gameState, ghostIndex, agent)
              This method uses the supplied ghost agent to determine what positions
              a ghost (ghostIndex) controlled by a particular agent (ghostAgent)
              will move to in the supplied gameState.  All ghosts
              must first be placed in the gameState using setGhostPositions above.

              The ghost agent you are meant to supply is self.ghostAgents[ghostIndex-1],
              but in this project all ghost agents are always the same.
        """
        newParticles = []
        for oldParticle in self.bubbles:
            newParticle = list(oldParticle) # A list of ghost positions

            # now loop through and update each entry in newParticle...

            # QUESTION 7

            i = 0

            while i < self.numGhosts: # go through each ghost
                dist = getPositionDistributionForGhost(setGhostPositions(gameState, newParticle), i, self.ghostAgents[i])
                newParticle[i] = util.sample(dist)

                # increase index
                i = i +1

            # END QUESTION 7
            newParticles.append(tuple(newParticle))

        self.bubbles = newParticles

    def getBeliefDistribution(self):
        # QUESTION 6 -----------------------------------------------------------
        distribubbles = util.Counter()  # belief distribution Counter object

        for i in self.bubbles:  # for each bubble
            distribubbles[i] = distribubbles[i] + 1  # increase by 1

        distribubbles.normalize()  # normalize otherwise the whole thing fucks up

        # return the distribution bc getter
        return distribubbles

# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()

def getPositionDistributionForGhost(gameState, ghostIndex, agent):
    """
    Returns the distribution over positions for a ghost, using the supplied gameState.
    """

    # index 0 is pacman, but the students think that index 0 is the first ghost.
    ghostPosition = gameState.getGhostPosition(ghostIndex+1)
    actionDist = agent.getDistribution(gameState)
    dist = util.Counter()
    for action, prob in actionDist.items():
        successorPosition = game.Actions.getSuccessor(ghostPosition, action)
        dist[successorPosition] = prob
    return dist

def setGhostPositions(gameState, ghostPositions):
    "Sets the position of all ghosts to the values in ghostPositionTuple."
    for index, pos in enumerate(ghostPositions):
        conf = game.Configuration(pos, game.Directions.STOP)
        gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
    return gameState

