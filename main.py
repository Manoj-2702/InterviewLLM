# import pandas as pd

# # Load the dataset
# data_path = 'Training.csv'
# data = pd.read_csv(data_path)
# # Sort the data by user_id in ascending order
# sorted_data = data.sort_values('user_id', ascending=True)
# # Count the frequency of each user_id
# user_id_counts = data['user_id'].value_counts()
# # Display the sorted data
# # print("Sorted Data by user_id:")
# # print(sorted_data.head())  # Displaying only the first few rows for brevity

# # Display the frequency of each user_id
# print("\nFrequency of each user_id:")
# print(user_id_counts)
# # Filter user_ids that appear more than once
# user_ids_more_than_one = user_id_counts[user_id_counts > 1]

# # Print the filtered user_ids and their counts
# print("User IDs with more than one occurrence:")
# print(user_ids_more_than_one)


# """
# Interviewer: That sounds comprehensive. Could you elaborate on the visually intuitive dashboards you designed for real-time insights at Cisco Systems? What tools and technologies did you use? 


# Interviewee: specifically node exporters, GMX exporters, and, process exporters. Sure. I used Grafana to set up the dashboards. And I also used, Prometheus, which was acting as the back end source for the Grafana dashboards. I set up mainly to monitor the availability and the spikes that are coming into the logs that we're receiving. So if there's a lag or somewhere in the transformers' logs, then we know that if the logs are missing or there's a lag, then there might be something wrong with the transformer. I also set up dashboards for the transformers availability. It itself so that we know that the EC 2 instances that are basically the transformers are up and running. Other than that, I also set up logs for the other types of metrics using different prompter queries like rates, irates, average, and some, So these were some ways that I set up the logs using the TxTag Grafana and Prometheus.
# """

# """
# Interviewer: Impressive work with Grafana and Prometheus. Moving on to your role at SociÃ©tÃ© GÃ©nÃ©rale, can you describe the micro-service based workflow logging and monitoring system you developed? What were the key components and technologies involved? 


# Interviewee: Sure. So, the task comprised of, setting up a logging system as well as a workflow monitoring system. Now in the workflow, there were 2 kinds of logs. There were the task logs and the system logs. The task logs were the airflow task logs itself, that I set up using s 3 bucket, so that all of the logs that are coming in from the workflows are directed to an s 3 bucket, and they're visible on the UI. The other part was the system logs. And in the system, you have the web server, the scheduler, and the worker components. So for that, I researched, some techniques or open source tools that I could use, and I ended up choosing, Fluent Bit for the log which acted as a log exporter. I, chose StatsD that had a built in support for metrics for our service. And I used telegraph to collect and process metrics from StatsD and then send it to Prometheus. So
# """


import pandas as pd

# Load the dataset
data_path = 'Training.csv'
data = pd.read_csv(data_path)

# Sort the data by user_id and question_id for consistency in operations
data.sort_values(by=['user_id', 'question_id'], inplace=True)
# Drop duplicates keeping the first occurrence
data = data.drop_duplicates(subset=['user_id', 'question_id'], keep='first')
# Define aggregation methods for concatenation
aggregations = {
    'question_cand_answer': '\n '.join,  # Concatenate answers
    'justification_judge': '\n'.join   # Concatenate justifications
}

# Group by 'user_id' and aggregate using defined methods
data = data.groupby('user_id').agg(aggregations).reset_index()
# Define the path for the new CSV
output_path = 'Processed_Training.csv'

# Save the DataFrame to a new CSV file
data.to_csv(output_path, index=False)

print(f"Processed data has been saved to {output_path}.")
