import matplotlib.pyplot as plt
plt.style.use('ggplot')


def pca_plot(pca, labels=None,figsize=(20,20)):
    components = pca.components_.T
    figure, ax = plt.subplots(1, figsize=figsize)
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))

    circle = plt.Circle((0, 0), 1, color='black', linestyle='--', fill=False)

    for i in range(0 ,components.shape[0]):
        ax.arrow(0, 0, components[i][0], components[i][1],
                  head_width=0.03, head_length=0.03,fc='b', ec='b' )
        
        if labels is None:
            plt.text(components[i,0]* 1.2, components[i,1] *  1.2, "Var"+str(i+1), ha = 'center', va = 'center')
        else:
            plt.text(components[i,0]*  1.2, components[i,1] * 1.2, labels[i], ha = 'center', va = 'center')

    ax.add_artist(circle)
    ax.set_xlabel(f'dim 1 ({round(pca.explained_variance_ratio_[0]*100, 1)}%)')
    ax.set_ylabel(f'dim 2 ({round(pca.explained_variance_ratio_[1]*100, 1)}%)')
    