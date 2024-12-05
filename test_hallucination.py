from openai import OpenAI
import numpy as np
from pydantic import BaseModel
import time

client = OpenAI(api_key="key")

class CheckEntailment(BaseModel):
    label: str

def check_entailment(fragment1: str, fragment2: str) -> bool:
    """check entailment"""
    messages = [
        {
            "role": "user",
            "content": f"""You have two responses from a large language model. 
                 Check if the meaning of one repsonse is entailed by the other, or if there is a contradiction. 
                 Return '0' if entailment. Return '1' if contradiction. 
                 Return only the label, without any explanation. 
                 \n Response1: \n {fragment1}\n\n Response2: \n {fragment2}""",
        }
    ]
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1,
        logprobs=True,
        top_logprobs=2,
        response_format=CheckEntailment,
    )
    entailment = False
    # print(completion.choices[0].logprobs.content[3].top_logprobs)
    for top_logprob in completion.choices[0].logprobs.content[3].top_logprobs:
        print(top_logprob.token, np.round(np.exp(top_logprob.logprob), 2))
        if "0" in top_logprob.token and np.exp(top_logprob.logprob) > 0.7:
            entailment = True
    return entailment


def calculate_entropy(probs):
    """
    Calculate the entropy
    """
    probs = np.array(probs)
    probs = probs / probs.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


some_tricky_questions = [
    "Which state does Alabama have its longest border with? Is it Florida or Tennessee?",
    "Who hosted the British Gameshow Countdown in 2007: a) Nick Hewer b) Richard Whiteley c) Jeff Stelling?",
    "Trivia question: Which Black Eyed Peas band member was the only one to host Saturday Night Live?",
    "What year in the 1980s were the FIS Alpine World Ski Championships hosted in Argentina?",
    "How many Brazilian numbers are there between 1-6?",
    "Which Israeli mathematician founded an online sequences repository in the 1970s?",
    "Write the 7 english words that have three consecutive double letters. No need to provide explanations, just say the words.",
    # adding two questions where it should not hallucinate
    "What is the capital of India?",
    "what is the full form of CPU?",
]


for question in some_tricky_questions:
    print("question", question)
    messages = [{"role": "user", "content": f"{question}"}]
    gpt_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1,
        logprobs=True,
        max_completion_tokens=60,
    )
    time.sleep(2)
    # get perplexity score using a low temperature response
    logprobs = [token.logprob for token in gpt_response.choices[0].logprobs.content]
    perplexity_score = np.round(np.exp(-np.mean(logprobs)), 2)
    # initialize clusters with the first response
    clusters = [[gpt_response.choices[0].message.content]]
    # generate some more responses using higher temperature and check entailment
    gpt_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        n=7,
        temperature=0.9,
        logprobs=True,
        max_completion_tokens=60,
    )
    time.sleep(2)
    # check entailment and form clusters
    responses = [choice.message.content for choice in gpt_response.choices]

    for response in responses[1:]:
        found_cluster = False
        for cluster in clusters:
            if check_entailment(cluster[0], response):
                cluster.append(response)
                found_cluster = True
                break
        if not found_cluster:
            clusters.append([response])
    cluster_probs = [len(cluster) / (len(responses) + 1) for cluster in clusters]
    discrete_entropy = calculate_entropy(cluster_probs)
    print("clusters", clusters)
    print("no of clusters", len(clusters))
    print("perplexity", perplexity_score)
    print("entropy", discrete_entropy)
