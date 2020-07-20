import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
data = pd.read_csv('survival_plot.csv')
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()


from lifelines.statistics import logrank_test
from lifelines.statistics import multivariate_logrank_test as mlrt
def calc_pvalue(T1, T2, E1, E2):
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    return results.p_value


'''OSS----------------------'''
#tertiles-------------------------------
T = data["OSSperiodmonths"]
E = 1- data["Osstatus"]

ax = plt.subplot(111)

l = (data["our_group_3"] == 0)
m = (data["our_group_3"] == 1)
h = (data["our_group_3"] == 2)
kmf.fit(T[l], event_observed=E[l], label="TILs low")
kmf.plot(ax=ax, show_censors=True, ci_show=False)
kmf.fit(T[m], event_observed=E[m], label="TILs mid")
kmf.plot(ax=ax, show_censors=True, ci_show=False)
kmf.fit(T[h], event_observed=E[h], label="TILs high")
kmf.plot(ax=ax, show_censors=True, ci_show=False)

plt.ylim(0, 1.05);
# plt.xlim(0, 100);
plt.title("Survial vs TIL");
plt.savefig('Our_OSS_Tertiles.png')
plt.close()

#medium--------------------------------------------------
ax = plt.subplot(111)

l = (data["our_group_2"] == 0)
h = (data["our_group_2"] == 1)
kmf.fit(T[l], event_observed=E[l], label="TILs low")
kmf.plot(ax=ax, show_censors=True, ci_show=False)
kmf.fit(T[h], event_observed=E[h], label="TILs high")
kmf.plot(ax=ax, show_censors=True, ci_show=False)

plt.ylim(0, 1.05);
# plt.xlim(0, 100);
plt.title("Survial vs TIL");
plt.savefig('Our_OSS_Medium.png')
plt.close()

#pathologist--------------------------------------------
ax = plt.subplot(111)

l = (data["path_group"] == 0)
m = (data["path_group"] == 1)
h = (data["path_group"] == 2)
kmf.fit(T[l], event_observed=E[l], label="TILs < 10%")
kmf.plot(ax=ax, show_censors=True, ci_show=False)
kmf.fit(T[m], event_observed=E[m], label="10% =< TILs < 60%")
kmf.plot(ax=ax, show_censors=True, ci_show=False)
kmf.fit(T[h], event_observed=E[h], label="TILs >= 60%")
kmf.plot(ax=ax, show_censors=True, ci_show=False)

plt.ylim(0, 1.05);
# plt.xlim(0, 100);
plt.title("Survial vs TIL");
plt.savefig('Path_OSS.png')
plt.close()

'''DSS-------------------------------'''
data = data[data['DSSperiodmonths'] != 999]
T = data["DSSperiodmonths"]
E = 1- data["DSSstatus"]

#tertile-------------------------------------------------------------
ax = plt.subplot(111)

test_result = mlrt(T, data["our_group_3"], E,)
print(test_result.p_value)
l = (data["our_group_3"] == 0)
m = (data["our_group_3"] == 1)
h = (data["our_group_3"] == 2)
kmf.fit(T[l], event_observed=E[l], label="TILs <10%")
kmf.plot(ax=ax, show_censors=True, ci_show=False)
kmf.fit(T[m], event_observed=E[m], label="TILs 10-59%")
kmf.plot(ax=ax, show_censors=True, ci_show=False)
kmf.fit(T[h], event_observed=E[h], label="TILs >=60%")
kmf.plot(ax=ax, show_censors=True, ci_show=False)

plt.ylim(0, 1.05);
# plt.xlim(0, 100);
plt.title("Survival Curve - Deep Multi-Task Learning");
plt.xlabel('months')
plt.savefig('Our_DSS_Tertile.png')
plt.close()

#medium-------------------------------------------------
ax = plt.subplot(111)

l = (data["our_group_2"] == 0)
h = (data["our_group_2"] == 1)
kmf.fit(T[l], event_observed=E[l], label="TILs low")
kmf.plot(ax=ax, show_censors=True, ci_show=False)
kmf.fit(T[h], event_observed=E[h], label="TILs high")
kmf.plot(ax=ax, show_censors=True, ci_show=False)

plt.ylim(0, 1.05);
# plt.xlim(0, 100);
plt.title("Survial vs TIL");
plt.savefig('Our_DSS_Medium.png')
plt.close()

#path-----------------------------------------------
ax = plt.subplot(111)

l = (data["path_group"] == 0)
m = (data["path_group"] == 1)
h = (data["path_group"] == 2)
kmf.fit(T[l], event_observed=E[l], label="TILs <10%")
kmf.plot(ax=ax, show_censors=True, ci_show=False)
kmf.fit(T[m], event_observed=E[m], label="TILs 10-59%")
kmf.plot(ax=ax, show_censors=True, ci_show=False)
kmf.fit(T[h], event_observed=E[h], label="TILs >=60%")
kmf.plot(ax=ax, show_censors=True, ci_show=False)

plt.ylim(0, 1.05);
# plt.xlim(0, 100);
plt.title("Survival Curve - Pathologist");
plt.xlabel('months')
plt.savefig('Path_DSS.png')
plt.close()
