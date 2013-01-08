"""
-----------------------
  Agent Configuration
-----------------------

Settings:

 - TeamName (string)
    The official name of your team. Names
    must be alpha-numeric only. Agents with
    invalid team names will not execute.

 - AgentFactory (string)
    The fully qualified name of the agent
    factory to execute.

 - AgentArgs (dict of string:string)
    Arguments to pass to the agent factory

 - NotifyList (list of strings)
    A list of email addresses to notify
    to when this agent competes.

 - Partners (list of strings)
    Group members who have contributed to
    this agent code and design.

"""

# Alpha-Numeric only
TeamName = 'MunchMunch'

# Filename.FactoryClassName (CASE-sensitive)
AgentFactory = 'munchAgents.MunchAgentFactory'

Partners = ['Sforzando, Rohan']

AgentArgs = {'first':'offense', 'second':'defense'}

NotifyList = []
