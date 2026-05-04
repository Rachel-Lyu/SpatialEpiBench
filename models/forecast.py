import torch

from . import metrics

class BaseTask:
    def __init__(self, prototype, model, dataset, lookback, horizon, device='cpu'):
        self.model = model
        self.prototype = prototype
        self.dataset = dataset
        self.device = device
        self.lookback = lookback
        self.horizon = horizon
        self.ahead = None
    
    def load_model(self, model):
        pass

    def load_dataset(self, dataset):
        pass

    def print_dataset(self, dataset = None):
        pass

class Forecast(BaseTask):
    """
    The Forecast class extends the BaseTask class, focusing on the training and evaluation of forecast models for 
    time-series prediction tasks. It includes functionalities specific to handling time-series data, especially 
    in settings that involve spatial-temporal dynamics. The class supports model initialization, training, evaluation, 
    and preprocessing, facilitating the application of various neural network architectures and configurations.
    """
    def __init__(self, prototype = None, model = None, dataset = None, lookback = None, horizon = None, device = 'cpu'):
        super().__init__(prototype, model, dataset, lookback, horizon, device)
        self.feat_mean = 0
        self.feat_std = 1
        self.device = device


    def train_model(self,
                    dataset=None,
                    config=None,
                    permute_dataset=False,
                    train_rate=0.6,
                    val_rate=0.2,
                    loss='mse', 
                    epochs=1000, 
                    batch_size=10,
                    lr=1e-3, 
                    weight_decay=0,
                    region_idx=None,
                    initialize=True, 
                    verbose=False, 
                    patience=100, 
                    device = None,
                    pretrained = None,
                    model_args={},
                    ):
        """
        Trains the forecast model using the provided dataset and configuration settings. It handles data splitting, model 
        initialization, and the training process, and also evaluates the model on the test set, reporting metrics such as 
        MAE and RMSE.
        """
        if device is not None:
            self.device = device
        # import ipdb; ipdb.set_trace()
        if config is not None:
            permute_dataset = config.permute
            train_rate = config.train_rate
            val_rate = config.val_rate
            loss = config.loss
            epochs=config.epochs
            batch_size=config.batch_size
            lr=config.lr
            initialize=config.initialize
            patience=config.patience

        if dataset is None:
            try:
                dataset = self.dataset
            except:
                raise RuntimeError("dataset not exists, please input dataset or use load_dataset() first!")
        else:
            self.dataset = dataset
        
        if not hasattr(self, "model"):
            raise RuntimeError("model not exists, please use load_model() to load model first!")
        
        self.region_index = region_idx
        self.train_split, self.val_split, self.test_split, self.adj = self.get_splits(self.dataset, train_rate, val_rate, region_idx, permute_dataset)
        if self.test_split['features'].numel() == 0:
            self.test_split = self.val_split

        try:
            self.target_mean, self.target_std = self.dataset.transforms.target_mean, dataset.transforms.target_std
        except:
            self.target_mean, self.target_std = 0, 1

       
        if pretrained is not None:
            # import ipdb; ipdb.set_trace()
            self.model = pretrained
            pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print('#params:',pytorch_total_params)
        else:
            if len(model_args) != 0:
                self.model = self.prototype(**model_args)
                pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print('#params:',pytorch_total_params)
            else:
                try:
                # initialize model
                    self.model = self.prototype(
                        num_nodes=self.adj.shape[0],
                        num_features=self.train_split['features'].shape[3],
                        num_timesteps_input=self.lookback,
                        num_timesteps_output=1,
                        device=self.device,
                        ).to(self.device)
                    pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    print('#params:',pytorch_total_params)
                except:
                    self.model = self.prototype(
                        num_features=self.train_split['features'].shape[2],
                        num_timesteps_input=self.lookback,
                        num_timesteps_output=1,
                        device=self.device,
                        ).to(self.device)
                    pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    print('#params:',pytorch_total_params)
        self.model = self.model.to(self.device)
        # import ipdb; ipdb.set_trace()
        # train
        try:
            self.model.fit(
                    train_input=self.train_split['features'], 
                    train_target=self.train_split['targets'], 
                    train_states = self.train_split['states'],
                    train_graph=self.adj, 
                    train_dynamic_graph=self.train_split['dynamic_graph'],
                    val_input=self.val_split['features'], 
                    val_target=self.val_split['targets'], 
                    val_states=self.val_split['states'],
                    val_graph=self.adj,
                    val_dynamic_graph=self.val_split['dynamic_graph'],
                    verbose=verbose,
                    batch_size=batch_size,
                    lr=lr,
                    weight_decay=weight_decay,
                    epochs=epochs,
                    loss=loss,
                    initialize=initialize,
                    patience=patience)

            # import ipdb; ipdb.set_trace()
            # evaluate
            self.test_graph = self.adj
            self.test_feature = self.test_split['features']
            self.test_target = self.test_split['targets']
            self.test_states = self.test_split['states']
            self.test_dynamic_graph = self.test_split['dynamic_graph']
            out = self.model.predict(feature=self.test_feature, 
                                    graph=self.test_graph, 
                                    states=self.test_states, 
                                    dynamic_graph=self.test_dynamic_graph
                                    ).reshape(self.test_target.shape)
            if type(out) is tuple:
                out = out[0]
            self.preds = out.detach().cpu()#*self.target_std+self.target_mean
            self.targets = self.test_target.detach().cpu()#*self.target_std+self.target_mean
            # metrics
            mse = metrics.get_MSE(self.preds, self.targets)
            mae = metrics.get_MAE(self.preds, self.targets)
            rmse = metrics.get_RMSE(self.preds, self.targets)
            print(f"Test MSE: {mse.item()}")
            print(f"Test MAE: {mae.item()}")
            print(f"Test RMSE: {rmse.item()}")
            
            return {"mse": mse.item(), "mae":mae.item(), "rmse":rmse.item(), "predictions": self.preds, "targets": self.targets}
        except Exception as e:
            print("Training failed!")
            print("Error message:", str(e))
            import traceback
            traceback.print_exc()
            # import ipdb; ipdb.set_trace()



    def evaluate_model(self,
                    model=None,
                    config=None,
                    features=None,
                    graph=None,
                    dynamic_graph=None,
                    norm=None,
                    states=None,
                    targets=None,
                    ):
        """
        Evaluates the trained model on a new dataset or using preloaded features and graphs. It outputs prediction accuracy 
        metrics such as MAE and RMSE for the forecasted values.
        """
        if model is None:
            if not hasattr(self, "model"):
                raise RuntimeError("model not exists, please use load_model() to load model first!")
            model = self.model
        # import ipdb; ipdb.set_trace()
        features = self.test_feature if features is None else features
        graph = self.test_graph if graph is None else graph
        states = self.test_states if states is None else states
        dynamic_graph = self.test_dynamic_graph if dynamic_graph is None else dynamic_graph
        targets = self.test_target if targets is None else targets

        # evaluate
        # import ipdb; ipdb.set_trace()
        with torch.no_grad():
            out = model.predict(feature=features, 
                                    graph=graph, 
                                    states=states, 
                                    dynamic_graph=dynamic_graph
                                    ).reshape(targets.shape)
        if type(out) is tuple:
            out = out[0]
        # import ipdb; ipdb.set_trace()

        self.preds = self.inverse_norm(out.detach().cpu(), norm)
        self.targets = self.inverse_norm(targets.detach().cpu(), norm)
        
        # metrics
        mse = metrics.get_MSE(self.preds, self.targets)
        mae = metrics.get_MAE(self.preds, self.targets)
        rmse = metrics.get_RMSE(self.preds, self.targets)
        print(f"Test MSE: {mse.item()}")
        print(f"Test MAE: {mae.item()}")
        print(f"Test RMSE: {rmse.item()}")
        
        return {"mse": mse.item(), "mae":mae.item(), "rmse":rmse.item(), "predictions":self.preds, "targets":self.targets}
    
    def inverse_norm(self, data, norm):
        mean = self.target_mean if norm is None else norm['mean']
        std = self.target_std if norm is None else norm['std']

        if type(std) is int:
            return data*std+mean
        else:
            return data*std.unsqueeze(-1)+mean.unsqueeze(-1)
    

    def get_splits(self, dataset=None, train_rate=0.6, val_rate=0.2, region_idx=None, permute=False):
        """
        Splits the provided dataset into training, validation, and testing sets based on specified rates. It also handles 
        preprocessing to normalize the data and prepare it for the model.
        """
        if dataset is None:
            try:
                dataset = self.dataset
            except:
                raise RuntimeError("dataset not exists, please use load_dataset() to load dataset first!")
            
        # preprocessing
        self.train_dataset, self.val_dataset, self.test_dataset = dataset.ganerate_splits(train_rate=train_rate, val_rate=val_rate)

        adj = self.train_dataset['graph']
        
        train_input, train_target, train_states, train_adj = dataset.generate_dataset(X=self.train_dataset['features'], 
                                                                                      Y=self.train_dataset['target'], 
                                                                                      states=self.train_dataset['states'],
                                                                                      dynamic_adj = self.train_dataset['dynamic_graph'],
                                                                                      lookback_window_size=self.lookback,
                                                                                      horizon=self.horizon,
                                                                                      permute=permute)
        val_input, val_target, val_states, val_adj = dataset.generate_dataset(X=self.val_dataset['features'], 
                                                                              Y=self.val_dataset['target'], 
                                                                              states=self.val_dataset['states'],
                                                                              dynamic_adj = self.val_dataset['dynamic_graph'],
                                                                              lookback_window_size=self.lookback, 
                                                                              horizon=self.horizon,
                                                                              permute=permute)
        test_input, test_target, test_states, test_adj = dataset.generate_dataset(X=self.test_dataset['features'], 
                                                                                  Y=self.test_dataset['target'], 
                                                                                  states=self.test_dataset['states'],
                                                                                  dynamic_adj = self.test_dataset['dynamic_graph'],
                                                                                  lookback_window_size=self.lookback, 
                                                                                  horizon=self.horizon,
                                                                                  permute=permute)
        if region_idx is not None:
            train_input = train_input[:,:,region_idx,:]
            val_input = val_input[:,:,region_idx,:]
            test_input = test_input[:,:,region_idx,:]

            train_target = train_target[:,:,region_idx]
            val_target = val_target[:,:,region_idx]
            test_target = test_target[:,:,region_idx]

            train_states = train_states[:,:,region_idx]
            val_states = val_states[:,:,region_idx]
            test_states = test_states[:,:,region_idx]

        return  {'features': train_input, 'targets': train_target, 'states': train_states, 'dynamic_graph': train_adj}, \
                {'features': val_input, 'targets': val_target, 'states': val_states, 'dynamic_graph': val_adj}, \
                {'features': test_input, 'targets': test_target, 'states': test_states, 'dynamic_graph': test_adj}, \
                adj

    def _slice_dataset_time(self, dataset, start_idx, end_idx):
        """Create a shallow time-sliced dataset for rolling retraining."""
        sliced = type(dataset)()
        sliced.x = dataset.x[start_idx:end_idx]
        sliced.y = None if getattr(dataset, "y", None) is None else dataset.y[start_idx:end_idx]
        sliced.states = None if getattr(dataset, "states", None) is None else dataset.states[start_idx:end_idx]
        sliced.dynamic_graph = (
            None
            if getattr(dataset, "dynamic_graph", None) is None
            else dataset.dynamic_graph[start_idx:end_idx]
        )
        sliced.graph = dataset.graph
        return sliced

    def run_rolling_retrain(
        self,
        dataset=None,
        horizons=(1,),
        retrain_every=7,
        retrain_train_length=None,
        first_target=None,
        permute_dataset=False,
        train_rate=0.6,
        val_rate=0.2,
        loss='mse',
        epochs=100,
        batch_size=10,
        lr=1e-3,
        weight_decay=0,
        region_idx=None,
        initialize=True,
        verbose=False,
        patience=100,
        device=None,
        model_args={},
    ):
        """
        Walk-forward retraining.
        For each retraining anchor and each horizon, retrain on a recent history slice
        and predict the next `retrain_every` target times when available.
        """
        if dataset is None:
            dataset = self.dataset

        if retrain_every < 1:
            raise ValueError(f"retrain_every must be >= 1, got {retrain_every}")
        if len(horizons) == 0:
            raise ValueError("horizons must be non-empty")

        n_total = dataset.x.shape[0]
        max_h = max(horizons)
        first_target = (self.lookback + max_h - 1) if first_target is None else first_target
        retrain_train_length = first_target if retrain_train_length is None else retrain_train_length

        if retrain_train_length < 1:
            raise ValueError(f"retrain_train_length must be >= 1, got {retrain_train_length}")

        # cache full windows per horizon for efficient target-index selection
        full_windows = {}
        for h in horizons:
            full_windows[h] = dataset.generate_dataset(
                X=dataset.x,
                Y=dataset.y,
                states=dataset.states,
                dynamic_adj=dataset.dynamic_graph,
                lookback_window_size=self.lookback,
                horizon=h,
                permute=permute_dataset,
            )

        out = {int(h): [] for h in horizons}
        original_horizon = self.horizon

        for start_idx in range(first_target, n_total, retrain_every):
            train_end = start_idx
            train_start = max(0, train_end - retrain_train_length)
            target_indices = list(range(start_idx, min(start_idx + retrain_every, n_total)))

            sliced_dataset = self._slice_dataset_time(dataset, train_start, train_end)

            for h in horizons:
                self.horizon = int(h)
                self.train_model(
                    dataset=sliced_dataset,
                    permute_dataset=permute_dataset,
                    train_rate=train_rate,
                    val_rate=val_rate,
                    loss=loss,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    weight_decay=weight_decay,
                    region_idx=region_idx,
                    initialize=initialize,
                    verbose=verbose,
                    patience=patience,
                    device=device,
                    model_args=model_args,
                )

                X_all, y_all, s_all, a_all = full_windows[h]
                offset = self.lookback + h - 1
                sample_ids = [t - offset for t in target_indices if 0 <= (t - offset) < X_all.shape[0]]
                valid_targets = [t for t in target_indices if 0 <= (t - offset) < X_all.shape[0]]
                if len(sample_ids) == 0:
                    continue

                x_eval = X_all[sample_ids]
                y_true = y_all[sample_ids]
                s_eval = None if s_all is None else s_all[sample_ids]
                a_eval = None if a_all is None else a_all[sample_ids]

                with torch.no_grad():
                    y_pred = self.model.predict(
                        feature=x_eval,
                        graph=dataset.graph,
                        states=s_eval,
                        dynamic_graph=a_eval,
                    )
                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]
                y_pred = y_pred.reshape(y_true.shape)

                out[int(h)].append(
                    {
                        "train_start": train_start,
                        "train_end": train_end,
                        "target_indices": valid_targets,
                        "y_true": y_true.detach().cpu(),
                        "y_pred": y_pred.detach().cpu(),
                    }
                )

        self.horizon = original_horizon
        return out