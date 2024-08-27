if __name__ == "__main__":
    import pickle
    from sklearn.model_selection import train_test_split

    TRAIN_GRAPHS = '/home/nbiescas/Desktop/CVC/CVC_internship/PKL_Graphs/V2_Kmeans_dis_ntresh_discrete/train_Kmeans_dis_ntresh_discrete.pkl'
    VAL_GRAPHS = '/home/nbiescas/Desktop/CVC/CVC_internship/PKL_Graphs/V2_Kmeans_dis_ntresh_discrete/val_Kmeans_dis_ntresh_discrete.pkl'
    with open(TRAIN_GRAPHS, 'rb') as f:
        train_graphs = pickle.load(f)
    
    train, val = train_test_split(train_graphs, test_size=0.2, random_state=42)
    with open(TRAIN_GRAPHS, 'wb') as path_train, open(VAL_GRAPHS, 'wb') as path_val:
        pickle.dump(train, path_train) # Train
        pickle.dump(val, path_val) # Val
