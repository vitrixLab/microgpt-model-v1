import random
from datetime import date, timedelta

random.seed(42)

def write_dataset(filename, num_samples=5000):
    templates = [
        # format: (list of question variants, answer function)
        ([
            "what is {hour}:{minute:02d} in words?",
            "how do you say {hour}:{minute:02d} in English?",
            "convert {hour}:{minute:02d} to words",
        ], lambda h, m: f"{number_words[h]} {number_words[m] if m != 0 else 'o\'clock'}"),
        
        ([
            "add {days} day(s) to {date_str}?",
            "what is {date_str} plus {days} days?",
            "{date_str} + {days} days = ?",
        ], lambda d, days: (d + timedelta(days=days)).isoformat()),
        
        ([
            "subtract {days} day(s) from {date_str}?",
            "what is {date_str} minus {days} days?",
            "{date_str} - {days} days = ?",
        ], lambda d, days: (d - timedelta(days=days)).isoformat()),
    ]

    number_words = {
        0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
        5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',
        10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen',
        14: 'fourteen', 15: 'fifteen', 16: 'sixteen', 17: 'seventeen',
        18: 'eighteen', 19: 'nineteen', 20: 'twenty', 30: 'thirty',
        40: 'forty', 50: 'fifty',
    }

    with open(filename, 'w') as f:
        for _ in range(num_samples):
            # Randomly pick a template family
            family = random.choice(templates)
            questions, answer_fn = family

            # Generate parameters
            if "hour" in questions[0]:  # time questions
                hour = random.randint(0, 23)
                minute = random.choice([0, 5, 15, 30, 45] + list(range(0,60)))
                q_template = random.choice(questions)
                question = q_template.format(hour=hour, minute=minute)
                answer = answer_fn(hour, minute)
            else:  # date arithmetic
                start_date = date(2020, 1, 1) + timedelta(days=random.randint(0, 1000))
                days = random.randint(1, 30)
                q_template = random.choice(questions)
                question = q_template.format(days=days, date_str=start_date.isoformat())
                answer = answer_fn(start_date, days)

            line = f"Q: {question} A: {answer}"
            f.write(line + '\n')

if __name__ == '__main__':
    write_dataset('dataset_time_date.txt', num_samples=5000)
    print("Dataset written to dataset_time_date.txt")