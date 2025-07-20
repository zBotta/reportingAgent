"""
A prompt generator class
"""

class PromptGenerator:

    def __init__(self,
                 what: str = None,
                 when: str = None,
                 where: str = None,
                 who: str = None,
                 how: str = None,
                 why: str = None,
                 contingency_actions: str = None):

        self.what = input('\nWhat has happend? \n[Describe event in few words] \n') if what is None else what
        self.when = input('\nWhen did the event happen? \n[Date & time of event occurrence and/or discovery] \n') if when is None else when
        self.where = input('\nWhere did the event happen? \n[Describe event location] \n') if where is None else where
        self.who = input('\nWho was involved? \n[Enumerate all involved persons and how they took part in event] \n') if who is None else who
        self.how = input('\nHow did the event happen?] \n') if how is None else how
        self.why = input('\nWhy did the event happen? \n[Describe root cause if known and/or ongoing investigations] \n') if why is None else why
        self.contingency_actions = input('\nWhich contingency actions have been taken? \n[Enumerate all actions taken subsequently to event] \n') if contingency_actions is None else contingency_actions


    def create_prompt(self, prompt_method: str = 'A'):
        '''Create prompts with various methods'''

        if prompt_method == 'A': # Simple instruction prompt
            self.prompt = f'Generate a text description on this event: \n\n[what: {self.what} \nwhen: {self.when} \nwhere: {self.where} \nwho: {self.who} \nhow: {self.how} \nwhy: {self.why} \ncontingency actions: {self.contingency_actions}]'
            return self.prompt

        if prompt_method == 'B': # Complex instruction prompt
            self.prompt = f'Generate a text description on this event: \n\n[what: {self.what} \nwhen: {self.when} \nwhere: {self.where} \nwho: {self.who} \nhow: {self.how} \nwhy: {self.why} \ncontingency actions: {self.contingency_actions}]\
            \nIt is important to focus on accuracy and coherence when generating the report so that the description content matches the information provided (what, when, where, who, how , why, contingency actions).\
            \nIf an information is not provided in (what, when, where, who, how , why, contingency actions), it must not be part of the generated text description.'
            return self.prompt

        if prompt_method == 'C': # Instruction prompt with example
            self.prompt = f'Generate a text description on this event: \n\n[what: {self.what} \nwhen: {self.when} \nwhere: {self.where} \nwho: {self.who} \nhow: {self.how} \nwhy: {self.why} \ncontingency actions: {self.contingency_actions}]\
            \n\nIt is important to focus on accuracy and coherence when generating the report so that the description content matches the information provided (what, when, where, who, how , why, contingency actions).\
            \nIf an information is not provided in (what, when, where, who, how , why, contingency actions), it must not be part of the generated text description.\
            \n\nHere is an example : \
            \nwith these information [what: Incorrect tablet count in bottle for Batch RX500 of Neurocet 50 mg \nwhen: July 2, 2025, 3:30 PM \nwhere: Bottle Packaging Line 2 \nwho: Erik Hansen (Packaging Operator, loaded wrong counting disk); Sarah Yoon (QA, identified deviation during AQL sampling) \nhow: Counting disk set for 60-count instead of 30-count \nwhy: Operator selected wrong format during changeover \ncontingency actions: Line stopped, 500 bottles segregated, rework initiated, operator retrained] \
            \nthe expected text description should look like "On July 2, 2025, at 3:30 PM, Erik Hansen loaded the wrong tablet counting disk during changeover on Bottle Packaging Line 2 for Batch RX500 of Neurocet 50 mg. Sarah Yoon from QA discovered the issue during AQL sampling. The line was stopped, 500 bottles were segregated, and rework and retraining were initiated."'
            return self.prompt

        else:
            raise ValueError('Invalid prompt method')