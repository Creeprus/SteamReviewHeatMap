
import json
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from ast import literal_eval
import os
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn import datasets
from sklearn.datasets import make_circles
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import make_moons
matplotlib.use('Agg')
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import seaborn as sns



def k_means(df: pd.DataFrame):  
    X = df[['price_final', 'positive_ratio']]
    kmeans = KMeans(random_state=42)
    kmeans.fit(X)
    df['cluster'] = kmeans.labels_
    print(df)
    kmeans_centroinds = kmeans.cluster_centers_
    plt.scatter(df['price_final'], df['positive_ratio'], c=kmeans.labels_)
    plt.xlabel('Финальная цена игры')
    plt.ylabel('Рейтинг')
    plt.title("Кластеризация", fontname="Times New Roman", fontweight="bold")
    plt.scatter(kmeans_centroinds[:, 0], kmeans_centroinds[:, 1], marker="o", color="black", s=30)
    #plt.tight_layout()
    plt.savefig(os.path.join('/home/pepegalord/Desktop/РЭУ/Big Data/Projects/BigDataGameRecommendations/graphs', '_k_means_clusterization'))
    plt.clf()


    
#svr алгоритм?
def db_scan(df: pd.DataFrame):
    df= df[['avg_hours', 'price_final']]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df)
    x_normal = normalize(x_scaled)
    x_normal = pd.DataFrame(x_normal)
    pca = PCA(n_components=2)
    x_principal = pca.fit_transform(x_normal)
    x_principal = pd.DataFrame(x_principal)
    db_scan = DBSCAN(eps=0.036, min_samples=4).fit(x_principal)
    print(df)
    plt.scatter(df['avg_hours'], df['price_final'], c=db_scan.labels_)
    plt.xlabel('Среднее время часов, которые игроки играли в игру')
    plt.ylabel('Финальная цена игры')
    plt.title("Кластеризация", fontname="Times New Roman", fontweight="bold")
    #plt.tight_layout()
    plt.savefig(os.path.join('/home/pepegalord/Desktop/РЭУ/Big Data/Projects/BigDataGameRecommendations/graphs', '_db_scan_clusterization'))
    plt.clf()


def export_corr_to_file(corr, filename):
    plot = sns.heatmap(corr)
    #plot.set_xticklabels(plot.get_xticklabels(), rotation=180)
    fig = plot.get_figure()
    fig.tight_layout()
    fig.savefig(os.path.join('/home/pepegalord/Desktop/РЭУ/Big Data/Projects/BigDataGameRecommendations/graphs', filename))
    plt.clf()

def get_price_change_ratings(df: pd.DataFrame):
    #Общая корреляция
    spearman_correlation = df.corr(method='spearman')   
    print(spearman_correlation)
    export_corr_to_file(spearman_correlation, 'all_correlation')
    #Корреляция изменения цены от отзыва
    df['price_diff'] = abs(df['price_final'] - df['price_original'])
    #df = df.drop(df[df['price_diff'] == 0.00].index)
    columns_to_keep = ['price_final', 'price_original', 'price_diff', 'positive_ratio', 'date_release']
    df = df[df.columns.intersection(columns_to_keep)]
    pearson_correlation = df.corr(method='pearson')   
    export_corr_to_file(pearson_correlation, 'price_diff_correlation')
    print(pearson_correlation)


def encode_model(df: pd.DataFrame):
    label = LabelEncoder()
    df['title'] = label.fit_transform(df['title'])
    df['date_release'] = label.fit_transform(df['date_release'])
    df['latest_review_date'] = label.fit_transform(df['latest_review_date'])
    df['earliest_review_date'] = label.fit_transform(df['earliest_review_date'])
    if 'tags' in df.columns:
        df['tags'] = label.fit_transform(df['tags'])
    return df


def correlate_genres(genres: list, true_genres: list, df: pd.DataFrame):
    columns_to_keep = ['price_final', 'price_original', 'price_diff', 'positive_ratio', 'date_release', 'tags', 'is_recommended_amount','not_recommended_amount', 'review_amount']
    df['review_amount'] = (df['is_recommended_amount']) + (df['not_recommended_amount'])
    df = df[df.columns.intersection(columns_to_keep)]
    for i in range(len(genres)-1):
        df_corr = df[df['tags'] == genres[i]]
        #print(df_corr)
        df_corr = df_corr.drop('tags', axis=1)
        #print(df_corr)
        correlation = df_corr.corr()
        correlation.dropna()
        print(correlation)
        print(true_genres[i], ' - ', genres[i])
        export_corr_to_file(correlation, filename='genres/'+str(true_genres[i]).replace('.','')+'_correlation')


if __name__ == '__main__':
    global file_path
    file_path = "GamesRecommendation.json"
    with open(file_path) as data_file:
        data = json.load(data_file)
    df = pd.json_normalize(data,  max_level=0)
    df = pd.DataFrame(data)
    df = df.drop('app_id', axis=1)
    df = df.drop('rating', axis=1)
    df_cluster = df
    df = df.explode('tags')
    df_cluster = df_cluster.drop('tags', axis=1)
    print(df_cluster)
    print(df_cluster.columns)

    print(df)
    print(df.columns)
    true_genres = list(dict.fromkeys(df['tags'].values.tolist()))
    true_genres = [x for x in true_genres if str(x) != 'nan']
    df = encode_model(df)
    k_means(encode_model(df_cluster))
    db_scan(encode_model(df_cluster))
    genres = list(dict.fromkeys(df['tags'].values.tolist()))
    genres = [x for x in genres if str(x) != 'nan']
    print(genres)
    print(df)
    get_price_change_ratings(df)
    correlate_genres(genres=genres, true_genres=true_genres, df=df)
