import pickle
import main

if __name__ == "__main__":
    """ -------------------------------- """
    """ PREPARE COMP 1 FILE FROM WEIGHTS """

    # Load trained weights (and extra objects needed) for our model
    c_path = rf'c_object_trained_on_train1test1.wtag.pkl'
    f_path = rf'f_object_trained_on_train1test1.wtag.pkl'
    weights_path = rf'weights_trained_on_train1test1.wtag.pkl'
    with open(c_path, 'rb') as file:
        c = pickle.load(file)
    with open(f_path, 'rb') as file:
        f = pickle.load(file)
    with open(weights_path, 'rb') as file:
        weights = pickle.load(file)

    # Run inference on competition 1 data file and write results to file according to .wtag format (described in HW1)
    main.inference(path_file_to_tag=r'comp1.words', path_result=r'comp_m1_308044296.wtag', weights=weights,
                   features_indices=f, class_statistics=c, beam=50)

    """ -------------------------------- """
    """ PREPARE COMP 2 FILE FROM WEIGHTS """

    # Load trained weights (and extra objects needed) for our model
    c_path = rf'c_object_trained_on_train2.wtag.pkl'
    f_path = rf'f_object_trained_on_train2.wtag.pkl'
    weights_path = rf'weights_trained_on_train2.wtag.pkl'
    with open(c_path, 'rb') as file:
        c = pickle.load(file)
    with open(f_path, 'rb') as file:
        f = pickle.load(file)
    with open(weights_path, 'rb') as file:
        weights = pickle.load(file)

    # Run inference on competition 2 data file and write results to file according to .wtag format (described in HW1)
    main.inference(path_file_to_tag=r'comp2.words', path_result=r'comp_m2_308044296.wtag', weights=weights,
                   features_indices=f, class_statistics=c, beam=50)
