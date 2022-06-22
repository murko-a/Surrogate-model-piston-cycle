def show_folds_fun(self):
    cnt = 1
    for train_index, test_index in self.kfold.split(self.X, self.y):
        print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
        cnt += 1