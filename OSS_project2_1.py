import pandas as pd

data = pd.read_csv("./2019_kbo_for_kaggle_v2.csv")

#problem 1
target_data = data[(data['year'] >= 2015) & (data['year'] <= 2018)] #2015년 ~ 2018년 까지만 추출
top_players = {}
for year in range(2015, 2019):
    top_players[year] = {
        'H' : target_data.nlargest(10, 'H'),      #nlargest함수를 이용해 각 부분별 상위 10명을 추출
        'avg' : target_data.nlargest(10, 'avg'),
        'HR' : target_data.nlargest(10, 'HR'),
        'OBP' : target_data.nlargest(10, 'OBP')
    }

for year, info in top_players.items():  #딕셔너리 안 딕셔너리 형태로 저장되어 있기 때문에 가독성을 위하여 변수를 나눠 받는 for문 두 개로 출력
    print(f"{year}년도")
    for cate, players in info.items():
        print(f"{cate} top 10:")
        print(players[['batter_name', cate]])
        print()
    print("===================")

print()
print()

#problem 2
#problem 1과 유사
target_data2 = data[(data['year'] == 2018)]
top_players2 = {}
position = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']
for pos in position:
    top_players2[pos] = target_data2[target_data2['tp'] == pos].nlargest(1, 'war')

for pos, info in top_players2.items():
    print(f"{pos}")
    print(info[['batter_name', 'war']])
    print("===================")

print()
print()

#problem 3
target_columns = ['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']
num_corr = data[target_columns].corrwith(data['salary']).abs() #corr함수를 이용하여 연봉과의 상관계수를 계산, abs는 절댓값
highest_corr = num_corr.idxmax()
print(f"연봉과 가장 관계가 높은 것은 {highest_corr}")