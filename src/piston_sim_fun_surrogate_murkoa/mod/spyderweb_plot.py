import plotly.graph_objects as go


def plot_spyderweb(self, performance_df):
    range_acc = 100
    range_mae = 0.15
    range_mse = 0.03
    range_rmse = 0.2
    range_r2 = 1.1
    range_time = 0.03

    ranges = [
        range_acc,
        range_mae,
        range_mse,
        range_rmse,
        range_r2,
        range_time]
    categories = ['Accuracy', 'mean(MAE)', 'mean(MSE)',
                  'mean(RMSE)', 'mean(R2)', 'time']
    fig = go.Figure()
    per_df = performance_df
    for mm in self.mdls_pf:
        per_df.loc[mm,
                   per_df.columns[0]] = per_df.loc[mm,
                                                   per_df.columns[0]] / range_acc
        per_df.loc[mm,
                   per_df.columns[1]] = per_df.loc[mm,
                                                   per_df.columns[1]] / range_mae
        per_df.loc[mm,
                   per_df.columns[3]] = per_df.loc[mm,
                                                   per_df.columns[3]] / range_mse
        per_df.loc[mm, per_df.columns[5]] = per_df.loc[mm,
                                                       per_df.columns[5]] / range_rmse
        per_df.loc[mm,
                   per_df.columns[7]] = per_df.loc[mm,
                                                   per_df.columns[7]] / range_r2
        per_df.loc[mm, per_df.columns[9]] = per_df.loc[mm,
                                                       per_df.columns[9]] / range_time
    for mm in self.mdls_pf:
        fig.add_trace(go.Scatterpolar(
            r=[per_df.loc[mm, performance_df.columns[0]],
               per_df.loc[mm, per_df.columns[1]],
               per_df.loc[mm, per_df.columns[3]],
               per_df.loc[mm, per_df.columns[5]],
               per_df.loc[mm, per_df.columns[7]],
               per_df.loc[mm, per_df.columns[9]]],
            theta=categories,
            fill='toself',
            name=mm
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title_text='Performance of models'
        )

    fig.show()
