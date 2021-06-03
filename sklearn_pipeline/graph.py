import matplotlib.pyplot as plt
from math import pi
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import traceback
import matplotlib.lines as mlines

request_headers="Exception class"

#main method
def polar_plot(df):
    try:
        #seperate numeric and categoric
        df_num = df.select_dtypes(include=['int64', 'float64'])
        df_txt = df.select_dtypes(include=['object'])
        # Label encoder for Categorical data
        le = LabelEncoder()
        for col in df_txt.columns:
            df_txt[col] = le.fit_transform(df_txt[col].astype(str))

        df = pd.concat([df_num, df_txt], axis=1)
        # normalization
        for col in df.columns:
            df[col]  = (df[col] - min(df[col])) / (max(df[col]) - min(df[col]))
        df=df.fillna(0.5)

        draw_polar_graph(df)
    except ValueError:
        print(request_headers + "ValueError from polar plot" + traceback.format_exc())
    except TypeError:
        print(request_headers + "TypeError from polar plot" + traceback.format_exc())
    except Exception:
        print(request_headers + "Error in polar plot" + traceback.format_exc())
        raise Exception("Error  in polar plot , check server logs for more information")



#plotting Graph
def draw_polar_graph(df):
    try:
        categories = list(df)[0:]
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], categories)
        ax.set_rlabel_position(0)
        plt.ylim(-0.1,1)

        # Plotting Graph
        rows=df.shape[0]
        for i in range(0,rows):
            values = df.iloc[[i]].values.flatten().tolist()
            values += values[:1]
            if(df.iloc[i]['outlier']):
                ax.plot(angles, values, linewidth=1,LineStyle="solid",color="red")
            else:
                ax.plot(angles, values, color="black", linewidth=1, linestyle=':')

        #adding labels
        normal = mlines.Line2D([], [], color='black', ls=":", label="Non-outlier")
        outliers = mlines.Line2D([], [], color='red', ls="-", label="outlier")
        # Add legend
        plt.legend(handles=[normal, outliers],loc='upper right', bbox_to_anchor=(0.1,0.1), prop={'size': 10})
        plt.show()

    except ValueError:
        print(request_headers + "ValueError from polar plot main while plotting a polar graph" + traceback.format_exc())
    except TypeError:
        print(request_headers + "TypeError from polar plot main while plotting a polar graph" + traceback.format_exc())
    except Exception:
        print(request_headers + "Error while plotting a polar graph in polar plot" + traceback.format_exc())
        raise Exception("Error while plotting a polar graph in polar plot , check server logs for more information")





