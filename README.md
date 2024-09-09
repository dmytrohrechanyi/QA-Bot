<h1 align="center">QA Bot</h1>


<p align="center">
  <a href="#dart-about">About</a> &#xa0; | &#xa0; 
  <a href="#sparkles-features">Features</a> &#xa0; | &#xa0;
  <a href="#rocket-technologies">Technologies</a> &#xa0; | &#xa0;
  <a href="#white_check_mark-requirements">Requirements</a> &#xa0; | &#xa0;
  <a href="#checkered_flag-starting">Starting</a> &#xa0;
</p>

<br>

## About ##

A boat that generates answers to questions using AI

## Features ##

Learning with new materials is possible.\
Generate answers to questions based on learned data.\
###### The current model is trained with the pdf in the specification folder.

## Technologies ##

The following tools were used in this project:

- [python](https://docs.python.org/3/)

## Requirements ##

Before starting this repository, you need to have [python](https://docs.python.org/3/)and [Visual Studio](https://learn.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2022) installed.
And you should download `mistral-7b-instruct-v0.1.Q4_K_M.gguf` from https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF and save the root folder

## Starting ##

```bash
# Clone this project
$ git clone https://github.com/dmytrohrechanyi/qa-bot

# Access
$ cd qa-bot

# Install dependencies
$ pip install -r requirements.txt
$  pip install langchain-community
$  pip install llama-cpp-python
# Run the project
$ streamlit run app.py

# The server will initialize in the <http://localhost:8501>
```