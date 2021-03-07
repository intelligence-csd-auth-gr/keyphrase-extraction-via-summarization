import numpy as np
import matplotlib.pyplot as plt
from data_statistics.nus_acm_semeval_paragraph_experiment.nus_acm_fulltext_statistics import acm_fulltext_statistics, nus_fulltext_statistics, semeval_fulltext_statistics,\
    acm_title_abstract_statistics, nus_title_abstract_statistics, semeval_title_abstract_statistics
from data_statistics.nus_acm_semeval_paragraph_experiment.first_3_paragraphs_statistics import acm_220_first_3_paragraphs_statistics,\
    acm_400_first_3_paragraphs_statistics, nus_220_first_3_paragraphs_statistics, nus_400_first_3_paragraphs_statistics,\
    semeval_220_first_3_paragraphs_statistics, semeval_400_first_3_paragraphs_statistics
from data_statistics.nus_acm_semeval_summary_experiment.nus_acm_summarized_statistics import acm_summarized_statistics, nus_summarized_statistics, semeval_summarized_statistics


# ======================================================================================================================
# Calculate test data (MUS, ACM) statistics
# ======================================================================================================================

# title + abstract statistics
acm_key_title_abstract = acm_title_abstract_statistics()
nus_key_title_abstract = nus_title_abstract_statistics()
semeval_key_title_abstract = semeval_title_abstract_statistics()


# fulltext statistics
acm_key_abstract, acm_total_keywords = acm_fulltext_statistics()
nus_key_abstract, nus_total_keywords = nus_fulltext_statistics()
semeval_key_abstract, semeval_total_keywords = semeval_fulltext_statistics()


# first 3 paragraphs statistics
acm_220_keywords_in_abstract = acm_220_first_3_paragraphs_statistics()
acm_400_keywords_in_abstract = acm_400_first_3_paragraphs_statistics()
nus_220_keywords_in_abstract = nus_220_first_3_paragraphs_statistics()
nus_400_keywords_in_abstract = nus_400_first_3_paragraphs_statistics()
semeval_220_keywords_in_abstract = semeval_220_first_3_paragraphs_statistics()
semeval_400_keywords_in_abstract = semeval_400_first_3_paragraphs_statistics()


# summary of abstract and full-text
acm_summarized_key_counts = acm_summarized_statistics()
nus_summarized_key_counts = nus_summarized_statistics()
semeval_summarized_key_counts = semeval_summarized_statistics()


print(acm_total_keywords)
print(nus_total_keywords)
print(semeval_total_keywords)


# ======================================================================================================================
# Calculate keyphrase percentage coverage
# ======================================================================================================================

# Barplot of percentages of keywords in title, abstract and combinations

# title + abstract
acm_title_abstract_key_abstract = 100 * np.round(acm_key_title_abstract / acm_total_keywords, 4)
nus_title_abstract_key_abstract = 100 * np.round(nus_key_title_abstract / nus_total_keywords, 4)
semeval_title_abstract_key_abstract = 100 * np.round(semeval_key_title_abstract / semeval_total_keywords, 4)


# fulltext statistics
acm_fulltext_key_abstract = 100 * np.round(acm_key_abstract / acm_total_keywords, 4)
nus_fulltext_key_abstract = 100 * np.round(nus_key_abstract / nus_total_keywords, 4)
semeval_fulltext_key_abstract = 100 * np.round(semeval_key_abstract / semeval_total_keywords, 4)


# first 3 paragraphs statistics
acm_220_key_abstract = 100 * np.round(acm_220_keywords_in_abstract / acm_total_keywords, 4)
acm_400_key_abstract = 100 * np.round(acm_400_keywords_in_abstract / acm_total_keywords, 4)

nus_220_key_abstract = 100 * np.round(nus_220_keywords_in_abstract / nus_total_keywords, 4)
nus_400_key_abstract = 100 * np.round(nus_400_keywords_in_abstract / nus_total_keywords, 4)

semeval_220_key_abstract = 100 * np.round(semeval_220_keywords_in_abstract / semeval_total_keywords, 4)
semeval_400_key_abstract = 100 * np.round(semeval_400_keywords_in_abstract / semeval_total_keywords, 4)


# summary of abstract and full-text
acm_summarized_key_percent = 100 * np.round(acm_summarized_key_counts / acm_total_keywords, 4)
nus_summarized_key_percent = 100 * np.round(nus_summarized_key_counts / nus_total_keywords, 4)
semeval_summarized_key_percent = 100 * np.round(semeval_summarized_key_counts / semeval_total_keywords, 4)


# ======================================================================================================================
# Visualize the counts of keywords in first 3 paragraphs
# ======================================================================================================================

acm_kp_percentages = [acm_220_key_abstract, acm_400_key_abstract]
nus_kp_percentages = [nus_220_key_abstract, nus_400_key_abstract]
semeval_kp_percentages = [semeval_220_key_abstract, semeval_400_key_abstract]
acm_kp_counts = [acm_220_keywords_in_abstract, acm_400_keywords_in_abstract]
nus_kp_counts = [nus_220_keywords_in_abstract, nus_400_keywords_in_abstract]
semeval_kp_counts = [semeval_220_keywords_in_abstract, semeval_400_keywords_in_abstract]
labels = ['3 first paragraphs \n paragraph length: 220', '3 first paragraphs \n paragraph length: 400']

x = np.arange(len(labels))  # the label locations
width = 0.26  # the width of the bars

fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
acm_rects = ax.bar(x - width, acm_kp_percentages, width, label='ACM')
nus_rects = ax.bar(x, nus_kp_percentages, width, label='NUS')
semeval_rects = ax.bar(x + width, semeval_kp_percentages, width, label='SemEval')

# add percentage symbol behind the values of the y axis
plt.gca().set_yticklabels(['{:.0f}%'.format(y_axis) for y_axis in plt.gca().get_yticks()])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_title("Keyphrase coverage percentage of the first 3 paragraphs"
             "\nTotal count of gold ACM keyphrases: {}"
             "\nTotal count of gold NUS keyphrases: {}"
             "\nTotal count of gold SemEval keyphrases: {}".format(acm_total_keywords, nus_total_keywords, semeval_total_keywords), pad=10)
ax.set_xlabel('Information source', labelpad=10)
ax.set_ylabel('Percentage of keyphrase coverage', labelpad=10)
ax.set_xticks(ticks=x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim([0, 100])
ax.legend(loc='upper right')  # 'upper left'


def autolabel(rects, kp_counts):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for index, rect in enumerate(rects):
        height = rect.get_height()
        plt.annotate('{:.2f}% \n {} KPs'.format(height, kp_counts[index]),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom',  fontsize=9)


autolabel(acm_rects, acm_kp_counts)
autolabel(nus_rects, nus_kp_counts)
autolabel(semeval_rects, semeval_kp_counts)

plt.tight_layout()

plt.show()

# plt.savefig('trainset_statistics.png')  # save the plot of model's loss per epoch to file


# ======================================================================================================================
# Visualize the counts of keywords in title + abstract and full-text
# ======================================================================================================================

acm_kp_percentages = [acm_title_abstract_key_abstract, acm_fulltext_key_abstract, acm_summarized_key_percent]
nus_kp_percentages = [nus_title_abstract_key_abstract, nus_fulltext_key_abstract, nus_summarized_key_percent]
semeval_kp_percentages = [semeval_title_abstract_key_abstract, semeval_fulltext_key_abstract, semeval_summarized_key_percent]
acm_kp_counts = [acm_key_title_abstract, acm_key_abstract, acm_summarized_key_counts]
nus_kp_counts = [nus_key_title_abstract, nus_key_abstract, nus_summarized_key_counts]
semeval_kp_counts = [semeval_key_title_abstract, semeval_key_abstract, semeval_summarized_key_counts]
labels = ['title + abstract', 'full-text', 'title \n+\n summarized documents']

x = np.arange(len(labels))  # the label locations
width = 0.26  # the width of the bars

fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
acm_rects = ax.bar(x - width, acm_kp_percentages, width, label='ACM')
nus_rects = ax.bar(x, nus_kp_percentages, width, label='NUS')
semeval_rects = ax.bar(x + width, semeval_kp_percentages, width, label='SemEval')

# add percentage symbol behind the values of the y axis
plt.gca().set_yticklabels(['{:.0f}%'.format(y_axis) for y_axis in plt.gca().get_yticks()])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_title("Keyphrase coverage percentage of the title + abstract, \nfull-text and summarized documents"
             "\nTotal count of gold ACM keyphrases: {}"
             "\nTotal count of gold NUS keyphrases: {}"
             "\nTotal count of gold SemEval keyphrases: {}".format(acm_total_keywords, nus_total_keywords, semeval_total_keywords), pad=10)
ax.set_xlabel('Information source', labelpad=10)
ax.set_ylabel('Percentage of keyphrase coverage', labelpad=10)
ax.set_xticks(ticks=x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim([0, 100])
ax.legend(loc='upper right')  # 'upper left'


def autolabel(rects, kp_counts):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for index, rect in enumerate(rects):
        height = rect.get_height()
        plt.annotate('{:.2f}% \n {} KPs'.format(height, kp_counts[index]),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom',  fontsize=9)


autolabel(acm_rects, acm_kp_counts)
autolabel(nus_rects, nus_kp_counts)
autolabel(semeval_rects, semeval_kp_counts)

plt.tight_layout()

plt.show()

# plt.savefig('trainset_statistics.png')  # save the plot of model's loss per epoch to file





'''
ACM title + abstract:  6518
ACM total keyphrases:  12296
ACM count of keywords in title + abstract:  0.5300910865322056

NUS title + abstract:  1264
NUS total keyphrases:  2458
NUS count of keywords in title + abstract:  0.5142392188771359

SemEval title + abstract:  1658
SemEval total keyphrases:  3778
SemEval count of keywords in title + abstract:  0.43885653785071466

ACM fulltext:  9079
ACM total keyphrases:  12296
ACM count of keywords in abstract:  0.738370201691607

NUS fulltext:  2157
NUS total keyphrases:  2458
NUS count of keywords in abstract:  0.8775427176566314

NUS fulltext:  3239
NUS total keyphrases:  3778
NUS count of keywords in abstract:  0.8573319216516675

ACM 220 - 3 first paragraphs:  7572
ACM 220 - total keyphrases:  12296
ACM 220 - count of keywords in abstract:  0.6158100195185426

ACM 400 - first 3 paragraphs:  8172
ACM 400 - total keyphrases:  12296
ACM 400 - count of keywords in abstract:  0.6646063760572544

NUS 220 - first 3 paragraphs:  1533
NUS 220 - total keyphrases:  2458
NUS 220 - count of keywords in abstract:  0.6236777868185517

NUS 400 - first 3 paragraphs:  1710
NUS 400 - total keyphrases:  2458
NUS 400 - count of keywords in abstract:  0.6956875508543532

SemEval 220 - first 3 paragraphs:  2197
SemEval 220 - total keyphrases:  3778
SemEval 220 - count of keywords in abstract:  0.5815246161990472

SemEval 400 - first 3 paragraphs:  2523
SemEval 400 - total keyphrases:  3778
SemEval 400 - count of keywords in abstract:  0.6678136580201165

ACM summarized:  7041
ACM summarized - total keyphrases:  12296
ACM summarized - count of keywords in abstract:  0.5726252439817827

NUS summary:  1415
NUS summary - total keyphrases:  2458
NUS summary - count of keywords in abstract:  0.5756712774613507

SemEval summarized:  1956
SemEval summarized - total keyphrases:  3778
SemEval summarized - count of keywords in abstract:  0.5177342509264161
'''
