import os
import csv

from line.agents.app import agent_rag

working_dir = "./line/data"

filenames = []
for filename in os.listdir(working_dir):
    filenames.append(filename)

for target in filenames:    
    filename = working_dir + "/" + target
    print(f"Target : [{filename}]")

    reviews = []
    with open(filename, mode="r", newline="", encoding="utf-16le") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        
        for row in csv_reader:
            content = row[11]
            rating = row[9]
            if content.strip() == "":
                continue
            reviews.append([rating, content])

    reviews_evaluated = []
    reviews_evaluated.append(["NO", "RATING", "CONTENT", "TYPE", "IS TECHNICAL", "EXPLANATION"])
    i = 1
    for r in reviews:
        prompt = f"Rating: {r[0]} Review: {r[1]}"
        response = agent_rag(prompt)
        reviews_evaluated.append([i, r[0], r[1], response[0], response[1], response[2]])
        i += 1

    output = filename + ".out.csv"
    with open(output, mode="w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(reviews_evaluated)
