#-*- coding:UTF-8 -*-
# author:jaylin
# time:17-7-8 下午4:10
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import math,datetime

train_user_values = '../data/SMPCUP2017_TrainingData_Task3.txt'
all_data = '../data/task3/all_data_task3.txt'
all_train_data = '../data/task3/all_train_data_task3.txt'
valid_user = '../data/valid_task3.txt'
all_train_0_1_data = '../data/task3/all_train_0_1_data_task3.txt'
all_train_0_data = '../data/task3/all_train_0_data_task3.txt'
all_train_1_data = '../data/task3/all_train_1_data_task3.txt'
valid_0_data = '../data/task3/valid_0_data.txt'
valid_1_data = '../data/task3/valid_1_data.txt'
train_predict_result = '../data/task3/train_predict_result.txt'
valid_predict_result = '../data/task3/valid_predict_result.txt'

def gen_0_1_classify_train_data(outfile,threshold=0.008):
    dataFrame = pd.read_csv(all_train_data,delimiter=' ',index_col=0)

    dataFrame_bigger = dataFrame[dataFrame['value']>threshold]
    dataFrame_bigger.to_csv(all_train_1_data,sep=',',index=True)
    dataFrame_bigger['value'] = 1
    dataFrame_smaller = dataFrame[dataFrame['value'] <= threshold]
    dataFrame_smaller.to_csv(all_train_0_data,sep=',',index=True)
    dataFrame_smaller['value'] = 0

    # print dataFrame_bigger, dataFrame_smaller

    dataFrame = pd.concat([dataFrame_bigger,dataFrame_smaller],axis=0)
    dataFrame.to_csv(all_train_0_1_data,sep=',',index=True)

# gen_0_1_classify_train_data(all_train_0_1_data)

# 相关系数计算与排序
def cal_correlation(filename):
    corrs = []
    data = pd.read_csv(filename, delimiter=',', index_col=0)
    for i in range(171):
        corr = data.iloc[:,i].corr(data.iloc[:,171])
        corrs.append(corr)
        print '第%s个特征的相关系数为:%s'%((i+1),corr)

    a = zip(corrs, range(len(corrs)))
    # print a
    a = sorted(a,key=lambda x: x[0],reverse=True)
    corrs = [x[0] for x in a]
    corrs_index_sort = [x[1] for x in a]
    print corrs
    print corrs_index_sort
    return corrs,corrs_index_sort

# 几个不同分类的相关系数对比
# # cal_correlation(all_train_0_data)
# [0.69788079444590267, 0.69221264765847468, 0.68154249336985262, 0.66180983376350255, 0.62843323275551721, 0.59276687097810354, 0.55259319298057807, 0.47530139696357027, 0.46946164319486994, 0.45780647776534378, 0.43940682107082324, 0.43879840513171114, 0.40830558160212055, 0.38708021193685838, 0.37320440520970566, 0.37139837995014535, 0.36904673309377728, 0.35748710271328232, 0.35716840723299864, 0.34778386260052258, 0.34636124050644934, 0.34015582203895955, 0.33846616635326882, 0.3320390674719812, 0.33132556555778181, 0.32376202915514535, 0.25325638039469073, 0.25028299757127087, 0.24188994614197507, 0.23679142149948135, 0.23522313184874535, 0.22263800909254794, 0.22172661011167633, 0.21865174706422186, 0.21207117606641721, 0.21192306404148134, 0.21190250825092755, 0.21113236644454922, 0.21035403641430089, 0.20359832027242175, 0.19560355175075628, 0.18939282381844019, 0.18790314816410839, 0.18109054394322222, 0.17824171349406012, 0.17234003598060701, 0.17100138953246602, 0.16890463274706558, 0.1675774246989232, 0.16659084653882342, 0.16620125559603069, 0.16492699620794987, 0.15385483183352563, 0.15329706782943839, 0.15160826448786358, 0.15006041590208058, 0.14796976446793614, 0.14776559694527214, 0.14638090359138678, 0.14522387808639164, 0.14421448280743182, 0.14263571422875895, 0.13905218054882668, 0.13724214886845218, 0.13710590604440606, 0.1357356527792572, 0.13471080935439475, 0.13399603997464277, 0.1337967778517119, 0.1205404916848678, 0.11610125962238439, 0.11557063944932315, 0.11393850920247428, 0.11274071016056759, 0.11274071016056759, 0.11274071016056759, 0.11274071016056759, 0.11274071016056759, 0.11274071016056759, 0.11274071016056741, 0.11044462668328728, 0.10689583973714353, 0.10458744001815989, 0.10376740608404621, 0.10319715720514366, 0.1026989617389743, 0.10088711275671383, 0.10012895633268237, 0.098526266175869404, 0.098083266361415872, 0.097918552438930356, 0.095227465835315939, 0.095025849591732076, 0.094859394316941126, 0.094338547505250586, 0.094202230059613534, 0.094202230059613409, 0.094202230059613257, 0.094091287821932962, 0.093449950840707965, 0.093412693367717001, 0.09232654298463229, 0.091286531173880789, 0.090876951981457607, 0.08972425926447665, 0.088486062322948575, 0.087570182602879645, 0.086999940561039116, 0.086718401094178429, 0.085827021099762146, 0.085384786141406696, 0.085291928879834292, 0.083965996231193982, 0.083818451146377718, 0.08277453455785895, 0.080463979388684104, 0.078148226869251214, 0.077161070207917473, 0.076878097492547065, 0.076618625788801298, 0.075345316516877162, 0.074744283310008486, 0.071988428725582768, 0.0698979931508317, 0.063256573436361097, 0.061080376958097184, 0.05904113882593566, 0.054681404137479905, 0.049585857099342052, 0.049232877323923542, 0.048728758444405863, 0.046872751239812813, 0.044102287625957227, 0.044056029972807068, 0.043691583266868755, 0.035691574497626187, 0.03295765885117892, 0.029868376522124884, 0.02649781273755494, 0.02461357175904301, 0.022312420433728979, 0.020830817816271895, 0.020483847618175139, 0.019775448600835521, 0.019312155372831489, 0.019021583085725099, 0.01816750677017653, 0.018167506770176478, 0.018167506770176416, 0.018167506770176371, 0.011274993477500326, 0.00089463839768544025, -0.011663395065106249, -0.012020537329083137, -0.012590730708185956, -0.022140031901102957, -0.023947887868170956, -0.03707754538606093, -0.040672930660679181, -0.05545712504543128, -0.064452711803720211, -0.06952044738233662, -0.084743819121145619, -0.086940078447464653, -0.0893243600992258, -0.090390804662620977, -0.090460792705022872, -0.091051715076711573, -0.1234912889695873, -0.12670527923420841, -0.19676333536844848]
# [97, 4, 20, 36, 21, 37, 5, 69, 96, 0, 53, 16, 1, 32, 112, 17, 162, 33, 122, 106, 142, 132, 111, 121, 152, 131, 102, 59, 11, 85, 84, 43, 27, 101, 75, 26, 91, 10, 65, 81, 108, 115, 125, 135, 38, 22, 99, 6, 42, 146, 55, 156, 68, 166, 126, 39, 49, 7, 116, 103, 161, 136, 23, 87, 110, 119, 71, 129, 139, 120, 24, 8, 40, 9, 25, 41, 57, 73, 89, 100, 123, 170, 140, 151, 133, 113, 52, 104, 29, 12, 28, 30, 14, 46, 141, 19, 35, 3, 67, 83, 95, 51, 105, 164, 61, 130, 98, 2, 18, 34, 70, 150, 154, 86, 47, 77, 160, 44, 31, 74, 107, 93, 15, 79, 63, 92, 94, 134, 114, 45, 124, 88, 90, 13, 80, 165, 155, 169, 128, 118, 76, 143, 137, 127, 117, 168, 138, 109, 148, 158, 64, 144, 147, 157, 167, 48, 62, 72, 56, 159, 60, 78, 149, 163, 153, 66, 50, 82, 54, 145, 58]
# cal_correlation(all_train_1_data)
# [0.94548695029934948, 0.92882790315532604, 0.91400421133475329, 0.90408066630644957, 0.90188076293226604, 0.87985786523761578, 0.86428723567712706, 0.86247766014991534, 0.83766562888206331, 0.82951513742391247, 0.82521416232334754, 0.81644129093764817, 0.80965293805409366, 0.8052574380917692, 0.80403875946449699, 0.8000847441414124, 0.79964897910210364, 0.78956344802359224, 0.7879011756223494, 0.77871747928984902, 0.7721560161857286, 0.77024459230892728, 0.73433150619525811, 0.72707139054323033, 0.72095885868299103, 0.71685871366547338, 0.71584828948203727, 0.71061661363101558, 0.70860831306979855, 0.70412121039543718, 0.69918187229852713, 0.69628206598226716, 0.68787651487721824, 0.68772717800951388, 0.68544366880336371, 0.66741570302977948, 0.66509848427683671, 0.66161253270752096, 0.65067411962471533, 0.63113543870680067, 0.60324845439384478, 0.59516340651326349, 0.5873114927301617, 0.57688207100902955, 0.56480607173233532, 0.55650671587682998, 0.55595206047196333, 0.55578284927389943, 0.55549943812384606, 0.55523903739121172, 0.55415483521735465, 0.55337583377304789, 0.55173812101434927, 0.55143043039238016, 0.55088320736577201, 0.54591487236245428, 0.54418046823150534, 0.54252229872506952, 0.52987267706831653, 0.52911412808017932, 0.52647730696140826, 0.52538993914505516, 0.52093423384297266, 0.51712979630953682, 0.51583728123677219, 0.51418111032876734, 0.50403605908540139, 0.47295596133748385, 0.44035588870073117, 0.43978607569215172, 0.42335937993796602, 0.4140634341051968, 0.40684326574491742, 0.40335081217437918, 0.39777302759460259, 0.3923305876094873, 0.38715016532323326, 0.38609091662723316, 0.37587989490417245, 0.35096953895149058, 0.34666465131130914, 0.31901879842446151, 0.31618706482136366, 0.31509987239956633, 0.31509145014050982, 0.28703062923011258, 0.28236176752843289, 0.28131493154427134, 0.27798032986398713, 0.27401898508893902, 0.26721471826819082, 0.26477996374350654, 0.26048846307759566, 0.24122528295610543, 0.19353516816772171, 0.16392432897257886, 0.16160675388884674, 0.15942001153461441, 0.15517976968891911, 0.15255392627813424, 0.14810048816251103, 0.14789603479252472, 0.14074384861251618, 0.14001282800426601, 0.1298264284407443, 0.12750968175937449, 0.1183906993848227, 0.10388353239726437, 0.09396502082489526, 0.092686763320484228, 0.092681448216334825, 0.091944904701131508, 0.090816480201316566, 0.090577552366637143, 0.089552604501906774, 0.087473438871121342, 0.086474017812097112, 0.086367391465850638, 0.083964713690339129, 0.070551884680195326, 0.065671912492294968, 0.064590448324090055, 0.064282881224289448, 0.063234236769838165, 0.062823638154021699, 0.055788381083855546, 0.051176549808714777, 0.050139722524350755, 0.050120031314951179, 0.050114315248068524, 0.04995716739443444, 0.048657448826050748, 0.047594632922532014, 0.04756326462401797, 0.044689017094834298, 0.044427734182967013, 0.041549215957297839, 0.037520535685196157, 0.035261476717062284, 0.033540127659329254, 0.033449101382187957, 0.032384399954361133, 0.030875039196343133, 0.026295448931998647, 0.024927776526996192, 0.023509628058420686, 0.016068085405669324, 0.016068085405669324, 0.016068085405669324, 0.016068085405669324, 0.016068085405669324, 0.016068085405669324, 0.014761542695724199, 0.013212793455587117, 0.012394759651673028, 0.01093750635239879, 0.0067811908653746108, 0.005967420755086608, -0.0018890618256760536, -0.0056767554141603721, -0.0090027848345088611, -0.014938467811719564, -0.024571564497015391, -0.025072649999549446, -0.038215211832897313, -0.042362324634614021, -0.052906784630232871, -0.054978438157528287, -0.06395384189745143, -0.08426057083529917, -0.091063290479206957]
# [37, 5, 1, 17, 85, 49, 21, 53, 33, 122, 146, 112, 81, 65, 132, 136, 116, 152, 126, 69, 156, 142, 166, 150, 96, 120, 0, 124, 16, 134, 154, 144, 170, 32, 114, 162, 160, 164, 130, 140, 19, 3, 35, 51, 97, 123, 133, 98, 34, 63, 2, 113, 18, 107, 4, 20, 143, 153, 83, 36, 7, 79, 67, 23, 39, 15, 31, 71, 95, 47, 87, 13, 148, 55, 158, 118, 29, 128, 66, 163, 61, 77, 168, 82, 138, 68, 22, 38, 84, 6, 93, 45, 99, 141, 50, 135, 121, 131, 111, 125, 106, 80, 115, 161, 108, 151, 52, 90, 11, 102, 129, 139, 165, 59, 119, 110, 169, 75, 159, 103, 70, 127, 137, 117, 109, 27, 105, 14, 104, 155, 30, 12, 28, 46, 86, 44, 43, 40, 24, 94, 42, 8, 149, 58, 100, 88, 9, 25, 41, 57, 73, 89, 91, 60, 26, 10, 74, 101, 78, 64, 147, 72, 157, 56, 167, 92, 76, 145, 62, 48, 54]

# 相关系数的比例类别0的 0.6:6 0.3:26 0.2:40
corrs = [0.69788079444590267, 0.69221264765847468, 0.68154249336985262, 0.66180983376350255, 0.62843323275551721, 0.59276687097810354, 0.55259319298057807, 0.47530139696357027, 0.46946164319486994, 0.45780647776534378, 0.43940682107082324, 0.43879840513171114, 0.40830558160212055, 0.38708021193685838, 0.37320440520970566, 0.37139837995014535, 0.36904673309377728, 0.35748710271328232, 0.35716840723299864, 0.34778386260052258, 0.34636124050644934, 0.34015582203895955, 0.33846616635326882, 0.3320390674719812, 0.33132556555778181, 0.32376202915514535, 0.25325638039469073, 0.25028299757127087, 0.24188994614197507, 0.23679142149948135, 0.23522313184874535, 0.22263800909254794, 0.22172661011167633, 0.21865174706422186, 0.21207117606641721, 0.21192306404148134, 0.21190250825092755, 0.21113236644454922, 0.21035403641430089, 0.20359832027242175, 0.19560355175075628, 0.18939282381844019, 0.18790314816410839, 0.18109054394322222, 0.17824171349406012, 0.17234003598060701, 0.17100138953246602, 0.16890463274706558, 0.1675774246989232, 0.16659084653882342, 0.16620125559603069, 0.16492699620794987, 0.15385483183352563, 0.15329706782943839, 0.15160826448786358, 0.15006041590208058, 0.14796976446793614, 0.14776559694527214, 0.14638090359138678, 0.14522387808639164, 0.14421448280743182, 0.14263571422875895, 0.13905218054882668, 0.13724214886845218, 0.13710590604440606, 0.1357356527792572, 0.13471080935439475, 0.13399603997464277, 0.1337967778517119, 0.1205404916848678, 0.11610125962238439, 0.11557063944932315, 0.11393850920247428, 0.11274071016056759, 0.11274071016056759, 0.11274071016056759, 0.11274071016056759, 0.11274071016056759, 0.11274071016056759, 0.11274071016056741, 0.11044462668328728, 0.10689583973714353, 0.10458744001815989, 0.10376740608404621, 0.10319715720514366, 0.1026989617389743, 0.10088711275671383, 0.10012895633268237, 0.098526266175869404, 0.098083266361415872, 0.097918552438930356, 0.095227465835315939, 0.095025849591732076, 0.094859394316941126, 0.094338547505250586, 0.094202230059613534, 0.094202230059613409, 0.094202230059613257, 0.094091287821932962, 0.093449950840707965, 0.093412693367717001, 0.09232654298463229, 0.091286531173880789, 0.090876951981457607, 0.08972425926447665, 0.088486062322948575, 0.087570182602879645, 0.086999940561039116, 0.086718401094178429, 0.085827021099762146, 0.085384786141406696, 0.085291928879834292, 0.083965996231193982, 0.083818451146377718, 0.08277453455785895, 0.080463979388684104, 0.078148226869251214, 0.077161070207917473, 0.076878097492547065, 0.076618625788801298, 0.075345316516877162, 0.074744283310008486, 0.071988428725582768, 0.0698979931508317, 0.063256573436361097, 0.061080376958097184, 0.05904113882593566, 0.054681404137479905, 0.049585857099342052, 0.049232877323923542, 0.048728758444405863, 0.046872751239812813, 0.044102287625957227, 0.044056029972807068, 0.043691583266868755, 0.035691574497626187, 0.03295765885117892, 0.029868376522124884, 0.02649781273755494, 0.02461357175904301, 0.022312420433728979, 0.020830817816271895, 0.020483847618175139, 0.019775448600835521, 0.019312155372831489, 0.019021583085725099, 0.01816750677017653, 0.018167506770176478, 0.018167506770176416, 0.018167506770176371, 0.011274993477500326, 0.00089463839768544025, -0.011663395065106249, -0.012020537329083137, -0.012590730708185956, -0.022140031901102957, -0.023947887868170956, -0.03707754538606093, -0.040672930660679181, -0.05545712504543128, -0.064452711803720211, -0.06952044738233662, -0.084743819121145619, -0.086940078447464653, -0.0893243600992258, -0.090390804662620977, -0.090460792705022872, -0.091051715076711573, -0.1234912889695873, -0.12670527923420841, -0.19676333536844848]
corrs = np.array(corrs)
print len(corrs[corrs>0.2])
# cal_correlation()
# 0/1分类的相关系数和排名为:
# [0.48582949966989308, 0.47557003901602296, 0.4746236644239693, 0.47436264232338132, 0.46856014695208964, 0.45031006964517506, 0.44947376927100979, 0.44501512582676073, 0.44432378028561897, 0.44308840585093845, 0.44291544746738548, 0.43688237558945775, 0.43503257561294917, 0.42221026520549321, 0.42095111113905626, 0.41886052593826367, 0.41828579241242814, 0.41804460431569052, 0.41601655756552908, 0.41314194241331376, 0.4058765347653237, 0.37879115713048966, 0.36078315532644967, 0.35417081272895773, 0.35096400276703443, 0.34894729249980133, 0.34793711236423069, 0.34698965256676623, 0.34639119530807955, 0.34521164884075944, 0.34145726667811743, 0.34093385955962474, 0.33885914838149339, 0.33785068640511717, 0.33025125146644224, 0.33010578158846082, 0.32956887397256418, 0.3295097181909995, 0.32842750752670424, 0.312234446707012, 0.31135995833711266, 0.30576107039588241, 0.30296619622632603, 0.30207046185056913, 0.3003898528107059, 0.29523570753630113, 0.29455787462974198, 0.2913920438349058, 0.28509375019299316, 0.28245895148113331, 0.27216525208421632, 0.25485329066674872, 0.25255512753658887, 0.24914578448449576, 0.24585423296670852, 0.24307033320614324, 0.23837333853301079, 0.23746964994888961, 0.23730389348782777, 0.23709092712303548, 0.23667728983488387, 0.23638073539816509, 0.23458738411942159, 0.23309186291058429, 0.23147763762717127, 0.22987651851925939, 0.22827591745688844, 0.22817814140947579, 0.22678673414834413, 0.22650434887507309, 0.22644506522185373, 0.22576842586230586, 0.22504939156241285, 0.22349789569944842, 0.22231684140407154, 0.22141650527010068, 0.21759066388824433, 0.21466400755837836, 0.20976222746248341, 0.20520421823749957, 0.19710403653463765, 0.19491414701527782, 0.18651744570137144, 0.18280572381046131, 0.18192936144454233, 0.17717815141509813, 0.17216718275625295, 0.17208952597332483, 0.17171852618684688, 0.16959346820149054, 0.16907024613996982, 0.16836708803494346, 0.16748847980817805, 0.1673057847966449, 0.16719038098646066, 0.1646045651976776, 0.16234312260524586, 0.16179642152715751, 0.16166390162498459, 0.16003892164454275, 0.16003892164454275, 0.16003892164454275, 0.16003892164454275, 0.16003892164454275, 0.16003892164454275, 0.14993129335267702, 0.14837915549363376, 0.14574985497664936, 0.14421507561449709, 0.14396667677342112, 0.14255357637027852, 0.14212930788962427, 0.13623067496690852, 0.13576387312687599, 0.13503534536970707, 0.13136748683548241, 0.12944755589884754, 0.128714051366491, 0.1245753302400211, 0.12338905592477924, 0.12334015295210093, 0.12142234783048024, 0.11868341092812616, 0.11413721353866124, 0.11161755288191862, 0.11115618788762462, 0.10895231894105649, 0.10828717069551141, 0.1080808633327312, 0.10769359984201367, 0.10642201021886798, 0.10587178336936233, 0.10252986302049695, 0.10206418182226806, 0.10101488428450868, 0.098018422130654759, 0.096120587088130205, 0.094431208200667013, 0.093801849889573172, 0.091901725911342394, 0.087928441080204464, 0.08691951560557884, 0.086813251793135371, 0.084770954250277961, 0.082127284267184783, 0.077898240705707666, 0.074507056653976486, 0.071301561823421525, 0.060205924781293441, 0.057649019580741312, 0.052840276478499856, 0.047526029315788101, 0.04655172621316471, 0.044202098906618179, 0.04255930105695413, 0.03742260574932349, 0.034201249719613226, 0.013037473643447801, 0.0072004747735559208, 0.005292371864364329, -0.00064290467633068289, -0.005628846325331038, -0.016386754049501046, -0.018946017944274057, -0.020299104508212991, -0.020363011323086384, -0.021430765059746611, -0.03408631839640397, -0.043660532894084718, -0.059263230046960567, -0.070274559774900339]
# [152, 112, 132, 122, 142, 49, 1, 33, 17, 65, 96, 0, 16, 69, 32, 162, 37, 53, 5, 21, 81, 85, 97, 4, 20, 156, 36, 116, 166, 170, 160, 136, 126, 71, 146, 150, 87, 120, 140, 130, 7, 23, 154, 164, 144, 39, 134, 124, 114, 55, 138, 102, 158, 168, 51, 79, 148, 83, 118, 67, 63, 128, 3, 35, 19, 31, 15, 47, 75, 95, 38, 22, 59, 84, 11, 6, 27, 99, 43, 93, 77, 61, 68, 91, 45, 101, 161, 10, 29, 133, 13, 26, 113, 107, 123, 34, 2, 18, 98, 9, 25, 41, 57, 73, 89, 131, 121, 143, 42, 153, 111, 135, 125, 106, 66, 80, 115, 141, 100, 108, 82, 103, 8, 24, 129, 139, 52, 119, 44, 28, 12, 110, 104, 163, 169, 50, 151, 64, 159, 40, 30, 14, 46, 105, 137, 127, 117, 109, 94, 149, 86, 58, 70, 48, 88, 165, 56, 90, 147, 74, 155, 60, 72, 157, 92, 78, 167, 54, 145, 76, 62]



# 训练二分类器
def train_bin_classify(max_round=50000,feature_list=None,feature_list_index=None):
    data = pd.read_csv(all_train_0_1_data, delimiter=',', index_col=0)
    if feature_list != None:
        feature_list.append('value')
        data = data[feature_list]
    if feature_list_index != None:
        feature_list_index.append(171)
        data = data.iloc[:, feature_list_index]
    print data.describe()
    train_index = data.index
    data = data.values
    x, y = np.split(data, (data.shape[1] - 1,), axis=1)  # 索引是后面的起点
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)
    test_num = y_test.shape[0]

    valid_userID = []
    with open('../data/valid_task3.txt', 'r') as f:
        ID = f.readline()
        while ID:
            if ID[-1] == '\n':
                valid_userID.append(ID[:-1])
            else:
                valid_userID.append(ID)
            ID = f.readline()

    user_data = pd.read_csv(all_data, delimiter=',', index_col=0)
    user_data = user_data.fillna(value=0)
    if feature_list != None:
        feature_list.remove('value')
        user_data = user_data[feature_list]
    if feature_list_index != None:
        feature_list_index.remove(171)
        user_data = user_data.iloc[:, feature_list_index]

    # 不写loc的话keyerror
    user_data = user_data.loc[valid_userID]
    user_data = user_data.values

    user_data_train = xgb.DMatrix(data[:, :-1])
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    valid_data = xgb.DMatrix(user_data, label=user_data[:, 0])
    watch_list = [(data_train, 'train'), (data_test, 'eval')]
    # 默认gbtree
    param = {'booster': 'gbtree', 'max_depth': 6, 'eta': 0.1, 'silent': 1, 'objective': 'multi:softmax',
             'num_class': 2}

    bst = xgb.train(param, data_train, num_boost_round=50000, evals=watch_list, early_stopping_rounds=1000)


    y_hat = bst.predict(data_test)
    # print y_hat
    result = y_test.reshape(1, -1) == y_hat
    print 'result', result
    print '正确率：\t', float(np.sum(result)) / len(y_hat)
    print 'END.....\n'

# [0.48582949966989308, 0.47557003901602296, 0.4746236644239693, 0.47436264232338132, 0.46856014695208964, 0.45031006964517506, 0.44947376927100979, 0.44501512582676073, 0.44432378028561897, 0.44308840585093845, 0.44291544746738548, 0.43688237558945775, 0.43503257561294917, 0.42221026520549321, 0.42095111113905626, 0.41886052593826367, 0.41828579241242814, 0.41804460431569052, 0.41601655756552908, 0.41314194241331376, 0.4058765347653237, 0.37879115713048966, 0.36078315532644967, 0.35417081272895773, 0.35096400276703443, 0.34894729249980133, 0.34793711236423069, 0.34698965256676623, 0.34639119530807955, 0.34521164884075944, 0.34145726667811743, 0.34093385955962474, 0.33885914838149339, 0.33785068640511717, 0.33025125146644224, 0.33010578158846082, 0.32956887397256418, 0.3295097181909995, 0.32842750752670424, 0.312234446707012, 0.31135995833711266, 0.30576107039588241, 0.30296619622632603, 0.30207046185056913, 0.3003898528107059, 0.29523570753630113, 0.29455787462974198, 0.2913920438349058, 0.28509375019299316, 0.28245895148113331, 0.27216525208421632, 0.25485329066674872, 0.25255512753658887, 0.24914578448449576, 0.24585423296670852, 0.24307033320614324, 0.23837333853301079, 0.23746964994888961, 0.23730389348782777, 0.23709092712303548, 0.23667728983488387, 0.23638073539816509, 0.23458738411942159, 0.23309186291058429, 0.23147763762717127, 0.22987651851925939, 0.22827591745688844, 0.22817814140947579, 0.22678673414834413, 0.22650434887507309, 0.22644506522185373, 0.22576842586230586, 0.22504939156241285, 0.22349789569944842, 0.22231684140407154, 0.22141650527010068, 0.21759066388824433, 0.21466400755837836, 0.20976222746248341, 0.20520421823749957, 0.19710403653463765, 0.19491414701527782, 0.18651744570137144, 0.18280572381046131, 0.18192936144454233, 0.17717815141509813, 0.17216718275625295, 0.17208952597332483, 0.17171852618684688, 0.16959346820149054, 0.16907024613996982, 0.16836708803494346, 0.16748847980817805, 0.1673057847966449, 0.16719038098646066, 0.1646045651976776, 0.16234312260524586, 0.16179642152715751, 0.16166390162498459, 0.16003892164454275, 0.16003892164454275, 0.16003892164454275, 0.16003892164454275, 0.16003892164454275, 0.16003892164454275, 0.14993129335267702, 0.14837915549363376, 0.14574985497664936, 0.14421507561449709, 0.14396667677342112, 0.14255357637027852, 0.14212930788962427, 0.13623067496690852, 0.13576387312687599, 0.13503534536970707, 0.13136748683548241, 0.12944755589884754, 0.128714051366491, 0.1245753302400211, 0.12338905592477924, 0.12334015295210093, 0.12142234783048024, 0.11868341092812616, 0.11413721353866124, 0.11161755288191862, 0.11115618788762462, 0.10895231894105649, 0.10828717069551141, 0.1080808633327312, 0.10769359984201367, 0.10642201021886798, 0.10587178336936233, 0.10252986302049695, 0.10206418182226806, 0.10101488428450868, 0.098018422130654759, 0.096120587088130205, 0.094431208200667013, 0.093801849889573172, 0.091901725911342394, 0.087928441080204464, 0.08691951560557884, 0.086813251793135371, 0.084770954250277961, 0.082127284267184783, 0.077898240705707666, 0.074507056653976486, 0.071301561823421525, 0.060205924781293441, 0.057649019580741312, 0.052840276478499856, 0.047526029315788101, 0.04655172621316471, 0.044202098906618179, 0.04255930105695413, 0.03742260574932349, 0.034201249719613226, 0.013037473643447801, 0.0072004747735559208, 0.005292371864364329, -0.00064290467633068289, -0.005628846325331038, -0.016386754049501046, -0.018946017944274057, -0.020299104508212991, -0.020363011323086384, -0.021430765059746611, -0.03408631839640397, -0.043660532894084718, -0.059263230046960567, -0.070274559774900339]
# [152, 112, 132, 122, 142, 49, 1, 33, 17, 65, 96, 0, 16, 69, 32, 162, 37, 53, 5, 21, 81, 85, 97, 4, 20, 156, 36, 116, 166, 170, 160, 136, 126, 71, 146, 150, 87, 120, 140, 130, 7, 23, 154, 164, 144, 39, 134, 124, 114, 55, 138, 102, 158, 168, 51, 79, 148, 83, 118, 67, 63, 128, 3, 35, 19, 31, 15, 47, 75, 95, 38, 22, 59, 84, 11, 6, 27, 99, 43, 93, 77, 61, 68, 91, 45, 101, 161, 10, 29, 133, 13, 26, 113, 107, 123, 34, 2, 18, 98, 9, 25, 41, 57, 73, 89, 131, 121, 143, 42, 153, 111, 135, 125, 106, 66, 80, 115, 141, 100, 108, 82, 103, 8, 24, 129, 139, 52, 119, 44, 28, 12, 110, 104, 163, 169, 50, 151, 64, 159, 40, 30, 14, 46, 105, 137, 127, 117, 109, 94, 149, 86, 58, 70, 48, 88, 165, 56, 90, 147, 74, 155, 60, 72, 157, 92, 78, 167, 54, 145, 76, 62]
# corrs = [0.48582949966989308, 0.47557003901602296, 0.4746236644239693, 0.47436264232338132, 0.46856014695208964, 0.45031006964517506, 0.44947376927100979, 0.44501512582676073, 0.44432378028561897, 0.44308840585093845, 0.44291544746738548, 0.43688237558945775, 0.43503257561294917, 0.42221026520549321, 0.42095111113905626, 0.41886052593826367, 0.41828579241242814, 0.41804460431569052, 0.41601655756552908, 0.41314194241331376, 0.4058765347653237, 0.37879115713048966, 0.36078315532644967, 0.35417081272895773, 0.35096400276703443, 0.34894729249980133, 0.34793711236423069, 0.34698965256676623, 0.34639119530807955, 0.34521164884075944, 0.34145726667811743, 0.34093385955962474, 0.33885914838149339, 0.33785068640511717, 0.33025125146644224, 0.33010578158846082, 0.32956887397256418, 0.3295097181909995, 0.32842750752670424, 0.312234446707012, 0.31135995833711266, 0.30576107039588241, 0.30296619622632603, 0.30207046185056913, 0.3003898528107059, 0.29523570753630113, 0.29455787462974198, 0.2913920438349058, 0.28509375019299316, 0.28245895148113331, 0.27216525208421632, 0.25485329066674872, 0.25255512753658887, 0.24914578448449576, 0.24585423296670852, 0.24307033320614324, 0.23837333853301079, 0.23746964994888961, 0.23730389348782777, 0.23709092712303548, 0.23667728983488387, 0.23638073539816509, 0.23458738411942159, 0.23309186291058429, 0.23147763762717127, 0.22987651851925939, 0.22827591745688844, 0.22817814140947579, 0.22678673414834413, 0.22650434887507309, 0.22644506522185373, 0.22576842586230586, 0.22504939156241285, 0.22349789569944842, 0.22231684140407154, 0.22141650527010068, 0.21759066388824433, 0.21466400755837836, 0.20976222746248341, 0.20520421823749957, 0.19710403653463765, 0.19491414701527782, 0.18651744570137144, 0.18280572381046131, 0.18192936144454233, 0.17717815141509813, 0.17216718275625295, 0.17208952597332483, 0.17171852618684688, 0.16959346820149054, 0.16907024613996982, 0.16836708803494346, 0.16748847980817805, 0.1673057847966449, 0.16719038098646066, 0.1646045651976776, 0.16234312260524586, 0.16179642152715751, 0.16166390162498459, 0.16003892164454275, 0.16003892164454275, 0.16003892164454275, 0.16003892164454275, 0.16003892164454275, 0.16003892164454275, 0.14993129335267702, 0.14837915549363376, 0.14574985497664936, 0.14421507561449709, 0.14396667677342112, 0.14255357637027852, 0.14212930788962427, 0.13623067496690852, 0.13576387312687599, 0.13503534536970707, 0.13136748683548241, 0.12944755589884754, 0.128714051366491, 0.1245753302400211, 0.12338905592477924, 0.12334015295210093, 0.12142234783048024, 0.11868341092812616, 0.11413721353866124, 0.11161755288191862, 0.11115618788762462, 0.10895231894105649, 0.10828717069551141, 0.1080808633327312, 0.10769359984201367, 0.10642201021886798, 0.10587178336936233, 0.10252986302049695, 0.10206418182226806, 0.10101488428450868, 0.098018422130654759, 0.096120587088130205, 0.094431208200667013, 0.093801849889573172, 0.091901725911342394, 0.087928441080204464, 0.08691951560557884, 0.086813251793135371, 0.084770954250277961, 0.082127284267184783, 0.077898240705707666, 0.074507056653976486, 0.071301561823421525, 0.060205924781293441, 0.057649019580741312, 0.052840276478499856, 0.047526029315788101, 0.04655172621316471, 0.044202098906618179, 0.04255930105695413, 0.03742260574932349, 0.034201249719613226, 0.013037473643447801, 0.0072004747735559208, 0.005292371864364329, -0.00064290467633068289, -0.005628846325331038, -0.016386754049501046, -0.018946017944274057, -0.020299104508212991, -0.020363011323086384, -0.021430765059746611, -0.03408631839640397, -0.043660532894084718, -0.059263230046960567, -0.070274559774900339]
# corrs = np.array(corrs)
# print len(corrs[corrs>0.2])
# # 0.4:21 0.3:45 0.2:80 0.1:135  80:0.906403940887 前80个特征
# features_list_index = [152, 112, 132, 122, 142, 49, 1, 33, 17, 65, 96, 0, 16, 69, 32, 162, 37, 53, 5, 21, 81, 85, 97, 4, 20, 156, 36, 116, 166, 170, 160, 136, 126, 71, 146, 150, 87, 120, 140, 130, 7, 23, 154, 164, 144, 39, 134, 124, 114, 55, 138, 102, 158, 168, 51, 79, 148, 83, 118, 67, 63, 128, 3, 35, 19, 31, 15, 47, 75, 95, 38, 22, 59, 84, 11, 6, 27, 99, 43, 93, 77, 61, 68, 91, 45, 101, 161, 10, 29, 133, 13, 26, 113, 107, 123, 34, 2, 18, 98, 9, 25, 41, 57, 73, 89, 131, 121, 143, 42, 153, 111, 135, 125, 106, 66, 80, 115, 141, 100, 108, 82, 103, 8, 24, 129, 139, 52, 119, 44, 28, 12, 110, 104, 163, 169, 50, 151, 64, 159, 40, 30, 14, 46, 105, 137, 127, 117, 109, 94, 149, 86, 58, 70, 48, 88, 165, 56, 90, 147, 74, 155, 60, 72, 157, 92, 78, 167, 54, 145, 76, 62]
# features_list_index = features_list_index[:20]
# train_bin_classify(feature_list_index=features_list_index)

# 训练二分类器并且对valid数据进行判断
def train_bin_classify_and_gen_valid(max_round=50000,feature_list=None,feature_list_index=None):
    data = pd.read_csv(all_train_0_1_data, delimiter=',', index_col=0)
    if feature_list != None:
        feature_list.append('value')
        data = data[feature_list]
    if feature_list_index != None:
        feature_list_index.append(171)
        data = data.iloc[:, feature_list_index]
    # print data.describe()

    data = data.values
    x, y = np.split(data, (data.shape[1] - 1,), axis=1)  # 索引是后面的起点
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)

    valid_userID = []
    with open(valid_user, 'r') as f:
        ID = f.readline()
        while ID:
            if ID[-1] == '\n':
                valid_userID.append(ID[:-1])
            else:
                valid_userID.append(ID)
            ID = f.readline()

    user_data = pd.read_csv(all_data, delimiter=',', index_col=0)
    user_data = user_data.fillna(value=0)
    # print user_data.describe()
    if feature_list != None:
        feature_list.remove('value')
        user_data = user_data[feature_list]
    if feature_list_index != None:
        feature_list_index.remove(171)
        user_data = user_data.iloc[:, feature_list_index]

    # 不写loc的话keyerror
    user_valid_data = user_data.loc[valid_userID]
    user_valid_data_values = user_valid_data.values

    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    valid_data = xgb.DMatrix(user_valid_data_values)
    watch_list = [(data_train, 'train'), (data_test, 'eval')]
    # 默认gbtree
    param = {'booster': 'gbtree', 'max_depth': 6, 'eta': 0.1, 'silent': 1, 'objective': 'multi:softmax',
             'num_class': 2}

    bst = xgb.train(param, data_train, num_boost_round=50000, evals=watch_list, early_stopping_rounds=1000)


    y_hat = bst.predict(data_test)
    result = y_test.reshape(1, -1) == y_hat
    print 'result', result
    print '正确率：\t', float(np.sum(result)) / len(y_hat)
    print 'END.....\n'

    # 预测valid的数据
    valid_result = bst.predict(valid_data)
    valid_result = [int(item) for item in valid_result]
    print valid_result
    print "预测结果长度:",len(valid_result)

    col_num = len(user_valid_data.columns)
    print col_num
    user_valid_data['type'] = valid_result

    valid_0_data_ = user_valid_data[user_valid_data.type==0].iloc[:,0:col_num]
    valid_1_data_ = user_valid_data[user_valid_data.type==1].iloc[:,0:col_num]

    valid_0_data_.to_csv(valid_0_data, sep=',', index=True)
    print "valid为0的数据集:"
    print valid_0_data_.describe()
    valid_1_data_.to_csv(valid_1_data,sep=',',index=True)
    print "valid为1的数据集:"
    print valid_1_data_.describe()


# train_bin_classify_and_gen_valid()

# 训练较大的数据集并且返回train和valid的预测值
def train_bigger_and_predict(max_round=50000,feature_list=None,feature_list_index=None):
    data = pd.read_csv(all_train_1_data, delimiter=',', index_col=0)
    if feature_list != None:
        feature_list.append('value')
        data = data[feature_list]
    if feature_list_index != None:
        feature_list_index.append(171)
        data = data.iloc[:, feature_list_index]

    train_index = data.index
    data = data.values
    print data.shape
    x, y = np.split(data, (data.shape[1] - 1,), axis=1)  # 索引是后面的起点
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)
    test_num = y_test.shape[0]



    valid_dataset = pd.read_csv(valid_1_data, delimiter=',', index_col=0)

    if feature_list != None:
        feature_list.remove('value')
        valid_dataset = valid_dataset[feature_list]
    if feature_list_index != None:
        feature_list_index.remove(171)
        valid_dataset = valid_dataset.iloc[:, feature_list_index]


    valid_data_ = valid_dataset.values
    valid_index = valid_dataset.index

    user_data_train = xgb.DMatrix(data[:, :-1])
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    valid_data = xgb.DMatrix(valid_data_)
    watch_list = [(data_train, 'train'), (data_test, 'eval')]
    # 默认gbtree
    param = {'booster': 'gbtree', 'max_depth': 6, 'eta': 0.1, 'silent': 1, 'objective': 'reg:logistic',
             'eval_metric': 'mae'}
    # reg:logistic0.30 和reg:linear效果很不好，误差0.648
    # mae 和 rmse 结果基本一样
    bst = xgb.train(param, data_train, num_boost_round=50000, evals=watch_list, early_stopping_rounds=1000)

    y_hat = bst.predict(data_test).reshape(test_num, )
    y_test = y_test.reshape(test_num, )

    y_max = []
    for i in range(test_num):
        y_max.append(y_test[i] if y_test[i] > y_hat[i] else y_hat[i])

    result = (np.abs(y_test - y_hat) / (np.array(y_max))).sum() / test_num
    print '模型的相对误差：', result



    y_predict = bst.predict(valid_data).reshape(valid_data_.shape[0], )
    y_train_hat = bst.predict(user_data_train).reshape(data.shape[0], )

    dataFrame_train = pd.DataFrame()
    dataFrame_valid = pd.DataFrame()
    dataFrame_train['id'] = train_index
    dataFrame_train['value'] = y_train_hat
    dataFrame_train.set_index(['id'],inplace=True)
    dataFrame_valid['id'] = valid_index
    dataFrame_valid['value'] = y_predict
    dataFrame_valid.set_index(['id'], inplace=True)

    print dataFrame_train
    # print dataFrame_valid
    # print dataFrame_train.describe()
    # print dataFrame_valid.describe()
    #
    # print 'END.....\n'
    return dataFrame_train,dataFrame_valid



# train_bigger_and_predict()


# 训练较小的数据集并且保存train和valid的预测值
def train_smaller_and_predict(max_round=50000,feature_list=None,feature_list_index=None):
    train_1_predict,valid_1_predict = train_bigger_and_predict()
    data = pd.read_csv(all_train_0_data, delimiter=',', index_col=0)
    print data
    if feature_list != None:
        feature_list.append('value')
        data = data[feature_list]
    if feature_list_index != None:
        feature_list_index.append(171)
        data = data.iloc[:, feature_list_index]

    train_index = data.index
    data = data.values
    print data.shape
    x, y = np.split(data, (data.shape[1] - 1,), axis=1)  # 索引是后面的起点
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)
    test_num = y_test.shape[0]

    valid_dataset = pd.read_csv(valid_0_data, delimiter=',', index_col=0)

    if feature_list != None:
        feature_list.remove('value')
        valid_dataset = valid_dataset[feature_list]
    if feature_list_index != None:
        feature_list_index.remove(171)
        valid_dataset = valid_dataset.iloc[:, feature_list_index]

    valid_data_ = valid_dataset.values
    valid_index = valid_dataset.index

    user_data_train = xgb.DMatrix(data[:, :-1])
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    valid_data = xgb.DMatrix(valid_data_)
    watch_list = [(data_train, 'train'), (data_test, 'eval')]
    # 默认gbtree
    param = {'booster': 'gbtree', 'max_depth': 6, 'eta': 0.1, 'silent': 1, 'objective': 'reg:logistic',
             'eval_metric': 'mae'}

    bst = xgb.train(param, data_train, num_boost_round=50000, evals=watch_list, early_stopping_rounds=1000)

    y_hat = bst.predict(data_test).reshape(test_num, )
    y_test = y_test.reshape(test_num, )

    y_max = []
    for i in range(test_num):
        y_max.append(y_test[i] if y_test[i] > y_hat[i] else y_hat[i])

    result = (np.abs(y_test - y_hat) / (np.array(y_max))).sum() / test_num
    print '模型的相对误差：', result

    y_predict = bst.predict(valid_data).reshape(valid_data_.shape[0], )
    y_train_hat = bst.predict(user_data_train).reshape(data.shape[0], )


    dataFrame_train = pd.DataFrame()
    dataFrame_valid = pd.DataFrame()
    dataFrame_train['id'] = train_index
    dataFrame_train['value'] = y_train_hat
    dataFrame_train.set_index(['id'], inplace=True)
    dataFrame_train = pd.concat([dataFrame_train,train_1_predict],axis=0)

    dataFrame_valid['id'] = valid_index
    dataFrame_valid['value'] = y_predict
    dataFrame_valid.set_index(['id'], inplace=True)
    dataFrame_valid = pd.concat([dataFrame_valid,valid_1_predict],axis=0)
    # dataFrame_valid.sort_index(axis=0)

    print dataFrame_train
    # print dataFrame_valid
    # print dataFrame_train.describe()
    # print dataFrame_valid.describe()

    dataFrame_train.to_csv(train_predict_result,sep=',',index=True)
    dataFrame_valid.to_csv(valid_predict_result,sep=',',index=True)

    print 'END.....\n'
    return dataFrame_train, dataFrame_valid



feature_list_idnex = [97, 4, 20, 36, 21, 37, 5, 69, 96, 0, 53, 16, 1, 32, 112, 17, 162, 33, 122, 106, 142, 132, 111, 121, 152, 131, 102, 59, 11, 85, 84, 43, 27, 101, 75, 26, 91, 10, 65, 81, 108, 115, 125, 135, 38, 22, 99, 6, 42, 146, 55, 156, 68, 166, 126, 39, 49, 7, 116, 103, 161, 136, 23, 87, 110, 119, 71, 129, 139, 120, 24, 8, 40, 9, 25, 41, 57, 73, 89, 100, 123, 170, 140, 151, 133, 113, 52, 104, 29, 12, 28, 30, 14, 46, 141, 19, 35, 3, 67, 83, 95, 51, 105, 164, 61, 130, 98, 2, 18, 34, 70, 150, 154, 86, 47, 77, 160, 44, 31, 74, 107, 93, 15, 79, 63, 92, 94, 134, 114, 45, 124, 88, 90, 13, 80, 165, 155, 169, 128, 118, 76, 143, 137, 127, 117, 168, 138, 109, 148, 158, 64, 144, 147, 157, 167, 48, 62, 72, 56, 159, 60, 78, 149, 163, 153, 66, 50, 82, 54, 145, 58]
train_smaller_and_predict(feature_list_index=feature_list_idnex[:40])