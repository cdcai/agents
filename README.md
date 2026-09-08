# Agents
[![Test Suite](https://github.com/cdcai/agents/actions/workflows/test.yaml/badge.svg)](https://github.com/cdcai/agents/actions/workflows/test.yaml)  

Sean Browning (NCEZID/OD Data Science Team)

## Background

I wanted to learn how to create and use language agents to solve complex problems. LangChain wasn't cutting it for me, so I made my own library from first principles to suit a few projects we're working on.

This package contains a few classes that can be used as building blocks for language agents and agentic systems. I plan to expand it with additional functionality as I need it, but keep a minimal footprint (ie. if you're already using openai and pydantic, this should bring no additional dependencies).

All code uses asyncio by design, and though I've tried to generalize as I can, I mostly built around OpenAI and specifically Azure OpenAI since that's what we are allowed to work with internally.

## Really? Another agent framework? Should I even use this library?

Maybe you *shouldn't* use this library.

After all, there are whole teams and companies of real software engineers working on similar frameworks that should suit your use-case:

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [smolagents](https://github.com/huggingface/smolagents)
- [AutoGen](https://github.com/microsoft/autogen)

By design, this library was designed to be python-ic, minimally invasive, and not require a lot of boilerplate to do what is essentially just string formatting and API request handling. It serves me well, but may not serve you.

## Installation

This isn't currently on pypi, so just use pip to install directly via GitHub:

```sh
pip install git+https://github.com/cdcai/multiagent.git
```

## Examples

| Example | Link |
| ---- | ---- |
| Taking output from one agent as input to another in a callback | [agent_with_callback.py](examples/agent_with_callback.py) |
| Getting structured output from agent / Text Prediction | [structured_prediction.py](examples/structured_prediction.py) |
| Batch processing large inputs over the same agent in parallel | [batch_processing.py](examples/batch_processing.py) |

## Public Domain and CC0 Notice

This repository constitutes a work of the United States Government and is not
subject to domestic copyright protection under 17 USC § 105. This repository is in
the public domain within the United States, and copyright and related rights in
the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
All contributions to this repository will be released under the CC0 dedication. By
submitting a pull request you are agreeing to comply with this waiver of
copyright interest.

See [LICENSE](LICENSE) for the complete CC0 1.0 legal code. Source code incorporated
from other projects retains its original license.

## Privacy Standard Notice
This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](DISCLAIMER.md)
and [Code of Conduct](code-of-conduct.md).
For more information about CDC's privacy policy, please visit [http://www.cdc.gov/other/privacy.html](https://www.cdc.gov/other/privacy.html).

## Contributing Standard Notice
Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo)
and submitting a pull request. (If you are new to GitHub, you might start with a
[basic tutorial](https://help.github.com/articles/set-up-git).) By contributing
to this project, you agree that your contribution will be released under the
[CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).

All comments, messages, pull requests, and other submissions received through
CDC including this GitHub page may be subject to applicable federal law, including but not limited to the Federal Records Act, and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice
This repository is not a source of government records, but is a copy to increase
collaboration and collaborative potential. All government records will be
published through the [CDC web site](http://www.cdc.gov).

## Additional Standard Notices
Please refer to [CDC's Template Repository](https://github.com/CDCgov/template) for more information about [contributing to this repository](https://github.com/CDCgov/template/blob/main/CONTRIBUTING.md), [public domain notices and disclaimers](https://github.com/CDCgov/template/blob/main/DISCLAIMER.md), and [code of conduct](https://github.com/CDCgov/template/blob/main/code-of-conduct.md).
