import argparse
import os
import textgrad as tg
from textgrad.engine.openai import AzureChatOpenAI
from dotenv import load_dotenv
import agents
load_dotenv()

# === Get creds, set env vars =================================
agents.openai_creds_ad()
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("GPT4_URL")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_trials", type=int, default=2)
    parser.add_argument("-m", "--model", type=str, default="edav-api-share-gpt4-api-nofilter")
    args = parser.parse_args()

    # === Set up engines =================
    eng = AzureChatOpenAI(args.model)
    eng_backprop = AzureChatOpenAI(args.model)
    tg.set_backward_engine(eng_backprop, override=True)

    model = tg.BlackboxLLM(eng)

    # === Run ====================================================
    question_string = ("If it takes 1 hour to dry 25 shirts under the sun, "
                    "how long will it take to dry 30 shirts under the sun? "
                    "Reason step by step")

    question = tg.Variable(question_string, 
                        role_description="question to the LLM", 
                        requires_grad=False)
    
    answer = model(question)

    answer.set_role_description("concise and accurate answer to the question")

    # Step 2: Define the loss function and the optimizer, just like in PyTorch! 
    # Here, we don't have SGD, but we have TGD (Textual Gradient Descent) 
    # that works with "textual gradients". 
    optimizer = tg.TGD(parameters=[answer])
    evaluation_instruction = (f"Here's a question: {question_string}. Here's the correct answer: 1 hour." 
                            "Evaluate any given answer to this question, "
                            "be smart, logical, and very critical. "
                            "Just provide concise feedback.")
                                

    # TextLoss is a natural-language specified loss function that describes 
    # how we want to evaluate the reasoning.
    loss_fn = tg.TextLoss(evaluation_instruction, engine=eng_backprop)

    for i in range(args.n_trials):
        print(f"====Round {i + 1} Answer====\n\n{answer}")

        loss = loss_fn(answer)
        loss.backward()
        optimizer.step()

        print(f"\nLoss: {loss.value}\n\n")
