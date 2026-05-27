# **CausalBench Dataset Data Card**

## **Dataset Description**
CausalBench is a comprehensive benchmark dataset designed to evaluate the causal reasoning capabilities of large language models (LLMs). It includes diverse tasks across three domains: code, math, and text, ensuring a robust assessment of causal inference abilities. Each causal scenario is presented with four different perspectives of questions: cause-to-effect, effect-to-cause, cause-to-effect with intervention, and effect-to-cause with intervention, along with their respective ground truths.

## **Dataset Summary**
- **Dataset Name**: CausalBench
- **Dataset Collaborator**: UCLA, JHU
- **Domains**: Code, Math, Text
- **Number of Samples**: 60,000 problems (40,000 text domain, 10000 code domain, 10000 math domain)
- **Languages**: English

## **Dataset Structure**

### **CausalBench_Code_Part.csv**:
- **Columns**: 
  - `Code`: Description of the coding scenario.
  - `Question Type`: Problem perspective.
  - `Question`: The code-based question description.
  - `Ground Truth`: The correct answer to the question.
  - `Explanation`: Explanation of the answer.
- **Problem Perspectives**:
  - Cause-to-Effect
  - Effect-to-Cause
  - Cause-to-Effect with Intervention
  - Effect-to-Cause with Intervention

### **CausalBench_Math_Part.csv**:
- **Columns**: 
  - `Mathematical Scenario`: Description of the mathematical scenario.
  - `Question Type`: Problem perspective.
  - `Question`: The math-based question description.
  - `Ground Truth`: The correct answer to the question.
  - `Explanation`: Explanation of the answer.
- **Problem Perspectives**:
  - Cause-to-Effect
  - Effect-to-Cause
  - Cause-to-Effect with Intervention
  - Effect-to-Cause with Intervention

### **CausalBench_Text_Part.csv**:
- **Columns**: 
  - `Scenario and Question`: Description of the textual scenario and question.
  - `Question Type`: Problem perspective.
  - `Ground Truth`: The correct answer to the text problem.
  - `Explanation`: Explanation of the answer.
- **Problem Perspectives**:
  - Cause-to-Effect
  - Effect-to-Cause
  - Cause-to-Effect with Intervention
  - Effect-to-Cause with Intervention

## **Usage**
CausalBench can be used for:
- Evaluating the causal reasoning capabilities of large language models.
- Conducting research on causal inference in AI.
- Developing and benchmarking new models for causal reasoning.

## **Data Collection**
The dataset was curated with a focus on diversity and complexity in causal reasoning tasks. Each domain (code, math, text) consists of scenarios, questions, answers, and detailed explanations of answers to ensure comprehensive coverage of causal reasoning aspects. Each causal scenario is assessed through four perspectives of questions to evaluate the model's understanding from multiple dimensions.

## **Data Construction Process**
The dataset construction process includes:
- **Manual Analysis and Generation**: Initial cases and ground truth generation based on causal graphs and conditional probabilities.
- **Scaling Up with LLMs**: Leveraging GPT-4 Turbo to expand the dataset across the three domains using few-shot prompting.
- **Quality Control**: Using a causal inference engine to check the answers and then human expert review to ensure the accuracy and reliability of the generated problems and ground truths.


## **Dataset Access**
The datasets can be accessed from the following links:
- [CausalBench_Code_Part.csv](https://huggingface.co/datasets/CCLV/CausalBench/blob/main/CausalBench_Code_Part.csv)
- [CausalBench_Math_Part.csv](https://huggingface.co/datasets/CCLV/CausalBench/blob/main/CausalBench_Math_Part.csv)
- [CausalBench_Text_Part.csv](https://huggingface.co/datasets/CCLV/CausalBench/blob/main/CausalBench_Text_Part.csv)


