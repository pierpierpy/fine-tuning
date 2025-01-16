template_llama2 = """ {% for message in messages %}
                        {% if message['role'] == 'user' %}
                                {{ bos_token + '[INST] ### Human: ' + message['content'] + ' [/INST]' }}
                        {% elif message['role'] == 'system' %}
                                {{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}
                        {% elif message['role'] == 'assistant' %}
                                {{ '\n ### Answer: '  + message['content'] + ' ' + eos_token }}
                        {% endif %}
                        {% endfor %}
                        """

template_mistral = """ {% for message in messages %}
                        {% if message['role'] == 'user' %}
                                {{ bos_token + '[INST] ### Human: ' + message['content'] + ' [/INST]' }}
                        {% elif message['role'] == 'system' %}
                                {{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}
                        {% elif message['role'] == 'assistant' %}
                                {{ '\n ### Answer: '  + message['content'] + ' ' + eos_token }}
                        {% endif %}
                        {% endfor %}
                        """
template_mistral_no_context = """{% for message in messages %}
                        {% if message['role'] == 'user' %}
                                {{ bos_token + '[INST] ### Human: ' + message['content'] + ' [/INST]' }}
                        
                        {% elif message['role'] == 'assistant' %}
                                {{ '\n ### Answer: '  + message['content'] + ' ' + eos_token }}
                        {% endif %}
                        {% endfor %}
                        """
template_llama2_no_context = """{% for message in messages %}
                        {% if message['role'] == 'user' %}
                                {{ bos_token + '[INST] ### Human: ' + message['content'] + ' [/INST]' }}
                        {% elif message['role'] == 'assistant' %}
                                {{ '\n ### Answer: '  + message['content'] + ' ' + eos_token }}
                        {% endif %}
                {% endfor %}
                        """


template_llama3 = """{{'<|begin_of_text|>'}}
                {% for message in messages %}
                        
                                {% if message['role'] == 'user' %}
                                        {{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}
                                {% elif message['role'] == 'system' %}
                                        {{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}
                                {% elif message['role'] == 'assistant' %}
                                        {{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}
                                {% endif %}
                        {% endfor %}"""
# https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/14
# mettere nel collator assistant e user
template_llama3 = """
{{'<|begin_of_text|>'}}
{% for message in messages %}
                        
                                {% if message['role'] == 'user' %}
                                        {{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}
                                {% elif message['role'] == 'system' %}
                                        {{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}
                                {% elif message['role'] == 'assistant' %}
                                        {{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}
                                {% endif %}
                        {% endfor %}"""


# use the following to see the collator at work. We see how the tokens from the user question and system message are ignored with the "ignore index" that is -100

# example = self.dataset["train"]["text"][0]
# tokenized_example = self.tokenizer(example)["input_ids"]
# self.datacollator.torch_call([tokenized_example]) --> takes the data in batch


# try this to show the ignored tokens:
# example = self.dataset["train"]["text"][0]
# tokenized_example = self.tokenizer(example)["input_ids"]
# labels = self.datacollator.torch_call([tokenized_example])["labels"][0] # take the tokens with the ignore index
# filtered_labels = [label for label in labels if label != -100] # filter the ignore index
# decoded_text = self.tokenizer.decode(filtered_labels) # decode the tokens of the assistant response
# decoded_text
