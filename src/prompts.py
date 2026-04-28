from __future__ import annotations


DEFAULT_GSM8K_INSTRUCTION_TEMPLATE = """\
Solve the following math word problem. Show your reasoning, then end with a line in the form "#### <answer>".

Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Reasoning: Natalia sold 48 clips in April. In May she sold half as many, so she sold 48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 clips.
#### 72

Question: Weng earns $12 an hour for babysitting. Yesterday, she babysat for 5 hours. How much did she earn?
Reasoning: Weng earns 12 dollars each hour. For 5 hours, she earned 12 * 5 = 60 dollars.
#### 60

Question: Betty has 24 marbles. She gives 8 marbles to her brother and then buys 5 more. How many marbles does Betty have now?
Reasoning: Betty starts with 24 marbles. After giving away 8, she has 24 - 8 = 16 marbles. Then she buys 5 more, so she has 16 + 5 = 21 marbles.
#### 21

Question: A restaurant has 9 tables. Each table seats 4 people. If 6 seats are empty, how many people are seated?
Reasoning: The restaurant has 9 * 4 = 36 total seats. Since 6 seats are empty, 36 - 6 = 30 people are seated.
#### 30

Question: {question}
Reasoning:"""
