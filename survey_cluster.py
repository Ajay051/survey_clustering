import os
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

class DataCleaningAndProcessing:
    """
    A class for cleaning, processing, and clustering survey response data from multiple files.
    This class reads survey response data from CSV files in a specified input folder,
    merges the data, performs cleaning operations, preprocesses the data, and clusters users.
    It then saves the cleaned and clustered data into separate Excel files.
    Attributes:
        None
    """
    def read_files():
        """
        Reads survey response data from CSV files, merges, cleans, preprocesses,
        clusters, and saves the data.
        This method prompts the user to input the folder path containing the CSV files.
        It then reads each CSV file, merges the dataframes, performs cleaning operations,
        preprocesses the data, clusters the users, and saves the cleaned and clustered data
        into separate Excel files.
        Returns:
            tuple: (tsw, ft)
        """
        dfs = {}
        no_need = [
            'Uu\n',
            # List of unnecessary entries
        ]
        directory = input("Enter the input folder path: ")
        if not os.path.isdir(directory):
            print("Invalid folder path!")
            sys.exit()

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                df = pd.read_csv(file_path)
                dfs[filename] = df

        survey_response = dfs['survey_response.csv']
        survey_response.replace({'group_id': {2: 'FASTRACK', 10: 'TITAN_SMART_WORLD'}}, inplace=True)
        sr_u = pd.merge(dfs['survey_response.csv'], dfs['user.csv'], on='user_id', how='inner')
        sr_u_sa = pd.merge(sr_u, dfs['survey_answer.csv'], left_on='id',
                           right_on='response_id', how='left')
        sq_sqt = pd.merge(dfs['survey_question.csv'], dfs['survey_question_type.csv'],
                          left_on='question_type_id', right_on='id')
        sq_sqt.drop(columns=['id_y'], inplace=True)
        sq_sqt.rename(columns={'id_x': 'id'}, inplace=True)
        sq_sqt_sqo = pd.merge(sq_sqt, dfs['survey_question_option.csv'],
                              left_on='id', right_on='question_id', how='inner')
        all_data = pd.merge(sr_u_sa, sq_sqt_sqo, on='question_id', how='left')
        all_data.rename(columns={'group_id': 'group_name'}, inplace=True)
        survey_name = all_data['survey_id_y'].str.split('_').str[0]
        name_survey=survey_name.iloc[0]
        all_data.drop(columns=['id_x', 'question_id', 'response_id', 'sequence',
                               'survey_id_y', 'question_type_id', 'order_x', 'order_y',
                               'created_time', 'start_value', 'end_value', 'increment',
                               'mandatory', 'type', 'value', 'id_y', 'display_text'], inplace=True)
        all_data.dropna(inplace=True)
        all_data = all_data[~all_data['answer'].isin(no_need)]

        # Add age and gender
        if 'dob' in all_data.columns:
            all_data['dob'] = pd.to_datetime(all_data['dob'])
            all_data['age'] = (pd.to_datetime('today') - all_data['dob']).dt.days // 365

        target_path = input("Give the output folder path: ")
        survey_name = all_data['survey_id_x'].str.split('_').str[0]
        name_survey = survey_name.iloc[0]
        all_data = all_data.drop(columns=['survey_id_x'])
        # all_data.to_excel(combined_file_path, index=False)

        # Saving Titan Smart World data to Excel
        tsw = all_data[all_data['group_name'] == 'TITAN_SMART_WORLD']
        # tsw.to_excel(combined_file_path, index=False)

        # Saving Fastrack data to Excel
        ft = all_data[all_data['group_name'] == 'FASTRACK']
        # ft.to_excel(combined_file_path, index=False)

        return tsw, ft, target_path,name_survey

class DataProcessor:
    def __init__(self, data):
        self.data = data
    def preprocess_and_cluster(self):
        if 'age' not in self.data.columns:
            print("Age column not found in the data. Continuing clustering without age column.")
            return None

        data_copy = self.data[['user_id', 'gender', 'question', 'answer', 'age']].copy()
        label_encoder = LabelEncoder()
        data_copy['answer_encoded'] = label_encoder.fit_transform(data_copy['answer'])
        data_copy['question_encoded'] = label_encoder.fit_transform(data_copy['question'])
        data_copy['gender'] = label_encoder.fit_transform(data_copy['gender'])
        grouped_data = data_copy.groupby('user_id')
        features_list = []

        for _, group in grouped_data:
            group_features = group[['age', 'answer_encoded']]
            features_list.append(group_features.values[0])
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_list)
        silhouette_scores = {}
        for num_clusters_range in range(2, 11):
            kmeans = KMeans(n_clusters=num_clusters_range,
                            init='k-means++', max_iter=300,
                            n_init=10, random_state=0)
            cluster_labels = kmeans.fit_predict(scaled_features)
            silhouette_avg = silhouette_score(scaled_features, cluster_labels)
            silhouette_scores[num_clusters_range] = silhouette_avg
        optimal_num_clusters = np.argmax(silhouette_scores) + 2
        print("Optimal number of clusters:", optimal_num_clusters)
        num_clusters = int(input("# of clusters: "))
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++',
                        max_iter=300, n_init=10, random_state=0)
        cluster_labels = kmeans.fit_predict(scaled_features)
        user_cluster_mapping = pd.DataFrame({'user_id': data_copy['user_id'].unique(),
                                             'cluster': cluster_labels})
        merged = pd.merge(data_copy,user_cluster_mapping,on='user_id')
        merged['gender'] = merged['gender'].replace({1: 'Male', 0: 'Female', 2: 'Others'})
        return merged

if __name__ == "__main__":
    tsw, ft, target_path, name_survey = DataCleaningAndProcessing.read_files()
    print("Processing Titan Smart World data...")
    tsw_processor = DataProcessor(tsw)
    tsw_clusters = tsw_processor.preprocess_and_cluster()
    tsw_filename = f"{name_survey.capitalize()}_Titan_cluster.xlsx"
    tsw_file_path = os.path.join(target_path, tsw_filename)
    tsw_clusters.to_excel(tsw_file_path,index=False)
    print(tsw_clusters)
    print("Processing Fastrack data...")
    ft_processor = DataProcessor(ft)
    ft_clusters = ft_processor.preprocess_and_cluster()
    ft_filename = f"{name_survey.capitalize()}_Fastrack_cluster.xlsx"
    ft_file_path = os.path.join(target_path, ft_filename)
    ft_clusters.to_excel(ft_file_path,index=False)
    print(ft_clusters)

