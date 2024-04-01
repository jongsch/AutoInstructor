import re
from enum import Enum
from textwrap import dedent
from typing import Literal

# from mediawiki import MediaWiki
import wikipedia
from mcts.base.base import BaseState
from mcts.searcher.mcts import MCTS
from pydantic import BaseModel, Field, model_validator, ValidationInfo

from client import conjure_model, client


# wikipedia = MediaWiki()


class Action(BaseModel):
    """
    A choice of action to take. Represents a branch in the tree of possible actions, leading to a new state.
    """
    type: Literal["think", "wiki"] = Field(..., description="The type of action to take.")
    text: str = Field(..., description="The text to execute as part of the action.")

    def execute(self, messages: list[dict[str, str]]):
        if self.type == "think":
            thought = conjure_model(client, messages + [{"role": "assistant",
                                                         "content": "Thinking about: " + self.text + "\nI will now generate the next thought."}],
                                    response_model=ThoughtContinuation)

            return thought.next_thought
        elif self.type == "wiki":
            wikipedia_search = conjure_model(client, messages + [{"role": "assistant",
                                                                  "content": "I have decided to search for this on wikipedia:\n" + self.text + "\nI will now generate a query."}],
                                             response_model=WikipediaSearch, validation_context={"messages": messages},
                                             max_retries=2)
            return f"Wikipedia search result for \"{wikipedia_search.query}\":\n{wikipedia_search.summary}"
        elif self.type == "chat":
            return input(self.text + "\n")


class ActionList(BaseModel):
    """A list of alternative actions to choose from. Must be parallel alternatives only, without sequential steps."""
    actions: list[Action] = Field(...,
                                  description="One or more possible next actions to choose from.")
    actions_eval: str = Field(..., description=dedent("""\
        Does the list contain any actions that depend on each other, and thus must run sequentially?
        Did you forget anything?
        Do any actions look extraneous or unlikely to make progress towards the goal?
        Do any look like they might require more information first?
    """))
    sequential: bool = Field(...,
                             description="Whether the list of actions contains any serial steps (instead of parallel branches only).")
    missing_actions: bool = Field(..., description="Whether any actions are missing from the list.")
    extraneous_actions: bool = Field(..., description="Whether any actions are extraneous.")

    @model_validator(mode="after")
    def check_actions(self):
        if self.sequential:
            raise ValueError(
                "The actions in the list must represent parallel branches ONLY, your answer included serial steps, please try again.")
        if self.missing_actions:
            raise ValueError("An action is missing from the list, please try again.")
        if self.extraneous_actions:
            raise ValueError("Some actions are extraneous to the list, please try again.")
        return self


class ThoughtContinuation(BaseModel):
    """
    A continuation of the thought process.
    """
    next_thought: str = Field(..., description="A new thought based on the input.")
    evaluation: str = Field(...,
                            description="An evaluation of the thought. Does it make progress, or is it a dead end? Is it a good idea?")
    good_idea: bool = Field(..., description="Whether the thought is a good idea.")

    @model_validator(mode="after")
    def check_continuation(self):
        if not self.good_idea:
            raise ValueError("The thought is not a good idea.")
        return self


#
# class WikipediaPageSelection(BaseModel):
#     """A selection of a wikipedia page from the search results."""
#     rationale: str = Field(..., description="Your rationale for selecting this page.")
#     page_name: str = Field(..., description="Which of the search results do you choose? Must match one of the results exactly.")
#     @model_validator(mode="after")
#     def validate_page(self, info: ValidationInfo):
#         search_results = info.context["search_results"]
#         if self.page_name not in search_results:
#             raise ValueError(f"The page name must be one of the search results: {search_results}")
#


def to_valid_identifier(s: str) -> str:
    """Transforms a string into a valid Python identifier for Enum members."""
    return re.sub(r'\W|^(?=\d)', '_', s)


def create_wikipedia_search_model_from_results(search_results: list[str]) -> "WikipediaPageSelection":
    PageNameEnum = Enum("PageNameEnum", {to_valid_identifier(choice): choice for choice in search_results})

    class WikipediaPageSelection(BaseModel):
        """A selection of a wikipedia page from the search results."""
        rationale: str = Field(..., description="Your rationale for selecting this page.")
        page_name: PageNameEnum = Field(...,
                                        description="Which of the search results do you choose? Must match one of the results exactly.")
        # @model_validator(mode="after")
        # def validate_page(self):
        #     if self.page_name not in search_results:
        #         raise ValueError(f"The page name must be one of the search results: {search_results}")

    return WikipediaPageSelection


class WikipediaContentExtraction(BaseModel):
    """A summary of the content of a Wikipedia page."""
    relevant_content: str = Field(..., description="A summary of the content on the wikipedia page relevant to the conversation.")
    extraction_eval: str = Field(..., description="An evaluation of the content extraction. Is it relevant to the goal? Is anything missing?")
    success: bool = Field(..., description="Whether the content extraction was successful.")

    @model_validator(mode="after")
    def check_extraction(self):
        if not self.success:
            raise ValueError("The content extraction was not successful.")
        return self


class WikipediaSearch(BaseModel):
    query: str = Field(..., description="The query to search for on Wikipedia.")
    summary: None

    @model_validator(mode="after")
    def search_wikipedia(self, info: ValidationInfo):
        search_results = wikipedia.search(self.query)
        messages = info.context["messages"]
        new_message1 = [{"role": "user", "content": "Select a page from the search results.: {search_results}"}]
        page_selection = conjure_model(client, messages + new_message1,
                                       response_model=create_wikipedia_search_model_from_results(search_results),
                                       validation_context={"search_results": search_results}, max_retries=2)
        new_message2 = [{"role": "assistant", "content": f"I have selected the page: \"{page_selection.page_name.value}\""}]
        new_message3 = [{"role": "user", "content": f"Page contents:\n{wikipedia.page(page_selection.page_name.value).content}\n\nPlease extract a summary of the relevant parts of the content."}]
        # self.summary = wikipedia.summary(page_selection.page_name.value, auto_suggest=False)
        self.summary = conjure_model(client, messages + new_message2 + new_message3, response_model=WikipediaContentExtraction, max_retries=3).relevant_content
        return self


#

class Evaluation(BaseModel):
    evaluation: str = Field(...,
                            description="How is the progress towards the goal? Is it taking any unnecessary detours?")
    is_terminal: bool = Field(...,
                              description="Whether the trajectory has terminated (true), or if there is more to be done (false). A trajectory that has gone so far off course that it must be abandoned should be marked as terminal (true)")
    score: int = Field(..., ge=0, le=5,
                       description="A score between 0 and 5, evaluating the trajectory in relation to goal progress and efficiency.")

    @model_validator(mode="after")
    def check_score(self):
        if self.is_terminal and self.score is None:
            raise ValueError("If the evaluation is terminal, a score must be provided.")
        return self


class LATSState(BaseState):
    def __init__(self, messages: list[dict[str, str]], evaluation: Evaluation | None = None):
        self.messages = messages
        if evaluation is None:
            self.evaluation: Evaluation = conjure_model(client, self.messages + [
                {"role": "user", "content": "Give an evaluation of the trajectory so far."}], response_model=Evaluation)
        else:
            self.evaluation = evaluation
        self._next_actions = None
        print(self.evaluation)

    def get_possible_actions(self) -> list[str]:
        if self._next_actions is None:
            self._next_actions: ActionList = conjure_model(client, self.messages + [
                {'role': 'user', 'content': "List possible next actions."}], response_model=ActionList)
        print(self._next_actions)
        return [action.model_dump_json() for action in self._next_actions.actions]

    def take_action(self, action_str: str) -> 'BaseState':
        action = Action.model_validate_json(json_data=action_str)
        print("taking action: ", action)
        observation = action.execute(self.messages)
        msg = dedent(f"""\
        Action taken: {action.type} | {action.text}
        Observation: {observation}
        """)
        print(msg)
        return LATSState(messages=self.messages + [{"role": "assistant", "content": msg}])

    def is_terminal(self) -> bool:
        return self.evaluation.is_terminal

    def get_reward(self) -> float:
        return self.evaluation.score

    def get_current_player(self) -> int:
        return 1


system_prompt = dedent("""\
You are an intelligent agent tasked with iteratively solving a goal.

We will be generating and executing a sequence of actions in order to solve a problem. Each action will be one of these types:

- Think some more (only do this if you are really stuck).
- Search Wikipedia for information.

You do not have any other actions available to you. You do not have internet access other than Wikipedia.
At each step, you will choose an action to take and provide the text needed to execute that action.
The steps you take will move you along a trajectory, as part of a larger monte carlo tree search.
At various points you will be asked to evaluate whether the current trajectory has terminated, and to score it.
""")
goal = "Find the NBA standings as of today."
print(goal)
messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": goal}, ]
initial_state = LATSState(messages, evaluation=Evaluation(evaluation="Initial state", is_terminal=False, score=0))

searcher = MCTS(iterationLimit=20)
bestAction = searcher.search(initial_state=initial_state)
print(bestAction)
