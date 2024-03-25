# Playing Soccer Games as Stochastic Games through Online Reinforcement Learning


$v_k(s,a)$ : the occurrence of $(s, a)$ at phase $k$
$$
vk(s,a) = 0 
$$
```
vk = np.zeros((len(states), len(actions)))
```

$n_k(s,a)$: the number of occurrences of $(s, a)$ before $k^{th}$ phase, if has not seen before set it to 1

<div style="text-align:center">
  <img src="https://latex.codecogs.com/svg.latex?n_k(s,a) = \max \left\{ \begin{array}{c} 1, \\ \sum_{\tau=1}^{t_k-1} \Pi_{(s_\tau, a_\tau) = (s,a)} \end{array} \right\}" alt="n_k(s,a) = max {1, sum}" />
</div>

$$
n_k(s,a) = max 
\left\{\begin{array}{c} 
1, \\
\sum_{\tau = 1}^{t_k-1} \Pi_{(s_\tau, a_\tau) = (s,a)}
\end{array} \right\}
$$

```
nk = np.zeros((len(states), len(actions)))
```


$n_k(s,a,s')$ :  use previous phase results to predict the next state, it looks at the number of s' transferred from $(s,a)$ from experience up to $k$ phase

$$
n_k(s,a, s') = 
\sum_{\tau =1 }^{t_k-1}{\Pi(s_\tau, a_\tau, s_{\tau+1})} \\
= (s,a,s')

$$

```
total_numbers = np.zeros((len(states), len(actions), len(states)))
```

$\hat{p_k}(s'| s,a)$:  transfer probability matrix

$$
\hat{p_k}(s'| s,a) = \frac{n_k(s,a,s')}{n_k(s,a)} 
\quad \forall s, a, s'
$$

- reform $n_k(s,a)$ to same shape as $n_k(s,a,s')$
- clip array within the range of `[1, None]`
```
p_hat = total_numbers/np.clip(nk.reshape(len(states), len(actions), 1), 1, None)	
```

$M_k$ : Model at phase $k$, is equal to a set of $\tilde{M}$, where $\tilde{M}$ is all probabilities transfer from $(s,a)$ for all s, a
$$
M_k = \left\{\begin{array}{c}
\tilde{M}: \forall s,a, \quad \tilde{p}(\cdot | s,a)  \in P_k(s,a)
\end{array} \right\}
$$

$P_k(s,a)$ : the union of confidence 1 and confidence 2



```
conf1 = np.sqrt((2*len(states)*np.log(1/delta)/len(states)))+ p_hat
conf2 = np.clip(np.sqrt(np.log(6/delta_1)/(2*len(states)))+p_hat, None, np.sqrt(2*p_hat*(1-p_hat)*np.log(6/delta_1)/len(states)) + 7*np.log(6/delta_1)/(3*(len(states)-1))+p_hat)
```


$MAXIMIN-EVI(M_k, \gamma_k)$

threshold of while loop: 
- a. the difference between maximum value at time $i$   and minimum value at time $i-1, \forall$ states
- b. the difference between minimum value at time $i$ and maximum value at time $i-1, \forall$ states
- terminates when $a-b <= (1- \alpha) * \gamma$ 

$\forall s, a$,  sum all the transfer probability with the previous value arrived at next state over all possible the next state $s'$ 
- **policy: choose Pk(s,a) such that the value is maximize**
```
Max_in = -1
for s in range(len(states)):
	for a in range(len(actions)):
		Sum = 0
	for next_s in range(len(states)):
		Sum += Pk[s][a][next_s] * v[i-1][next_s]
	if Sum >= Max_in: 
		Max_in = Sum
		Max_in_Pk_s = s
		Max_in_Pk_a = a
```

for each state $s$, calculate the max value then **select an action if value for the state interested has increased**

```
for s in range(len(states)):
	Max_out = -1
	for a in range(len(actions)):
		val = total_rewards[s,a] + Max_in
		if val >= Max_out:
			Max_out = val
			if st_i == s:
				Max_out_a = a
				choose_action = Max_out_a
	v[i][s] = (1-alpha) * val + alpha * v[i-1][s]
```



return action: in case the value stored is not sufficient and does not enter while loop, directly returns a random action
```
 try:
        return actions[choose_action]
except: # 一開始可能不會進到前面的while迴圈，所以直接回傳random的action
        print('!')
        action_i = random.randint(0, len(actions)-1)
        return actions[action_i]
```

##  研究動機與問題
強化學習（Reinforcement Learning）指的是agent與環境不斷的互動，並從中學習以取得最大的預期報酬。隨著研究發展，強化學習被應用在許多領域上，包括電子競技，機器人控制，工業自動化。近年來，強化學習在許多電玩遊戲中展現了能與頂尖選手競爭的能力，例如戰勝世界棋王的AlphaGo、 1V1 單挑戰勝《Dota 2》前世界冠軍的OpenAI。  
調查指出，足球的估計球迷人數為35億，居所有運動之冠，其商業價值更是高達約250億美元 。足球的熱門也帶動了相關產業的發展，例如《FIFA》系列是足球模擬器遊戲的一種，是目前世界上最暢銷的體育電玩遊戲系列。Google團隊在2019年推出了Google Research Football (Kurach et al., 2019)[1]，以熱門足球電玩遊戲為基礎，提供agent在基於物理的3D模擬器下進行足球比賽並學習的環境，其中agent可以控制自己隊伍的一個或所有足球運動員，學習如何在他們之間傳球，並設法克服對手的防守，以求得進球。以Google Research Football 為強化學習環境，本研究將會聚焦在足球賽局的應用，並利用隨機博弈模擬兩支隊伍在球場上的互動。  
在現實世界中有許多應用皆可以利用多人隨機博弈(multi-agent stochastic game)作為模型，例如選舉、西洋棋與足球。賽局的狀態會因參與者選擇特定行動而獲得相對應的報酬並會影響到下一個隨機狀態，而參與者得到的總報酬常用各階段收益平均值的下極限計算。要在足球遊戲中獲勝需要在每個狀態下做出最有利的決策，例如:球靠近球門時守門員須在正確的位置防守、傳球時的隊伍要有效布局以利進攻等，因此，過程是動態且伴隨著高度不確定性。本研究模型設定為online，意即只控制其中一個隊伍甲，假設隊伍中每一位選手的能力相同，並與對手隊伍乙比賽。模型目標是在可防守的狀況下盡量踢進對方的球門以最大化報酬。  

## 初步研究 
### 強化學習數學定義
設定模型 $M\ =(S,A,r,p)$ ，其中 $S$ 為狀態空間， $A$ 為兩個玩家的動作空間， $r$ 為報酬， $p$ 為轉移機率。在時間點 $t$ 時，兩個玩家在狀態 $s_t$，他們採取的動作為 $at=(a^1_t , a^2_t)$ ，學習者獲得的報酬為 $r_t=r(s_t, a^1_t , a^2_t)\ \in[0,\ 1]$ ，由狀態 $s_t$ 到達狀態 $s_{t+1}$ 的轉移機率為 $p(\cdot|\ s_t, a^1_t , a^2_t)$ 。此外，學習者是從策略 $\pi_1$ 取得動作 $a_1$ ，對手是從策略 $\pi_2$ 取得動作 $a_2$ ， $(\pi1,\pi2)$ 可縮寫為 $\pi$ ， $(a_1, a_2)$ 可縮寫為 $a$ 。在遊戲共進行T步下，定義以下報酬與regret: 
	T步總報酬:
	
<img src="https://i.imgur.com/NC8i3i6.png" width="23%"/>
	平均報酬:
	
<img src="https://i.imgur.com/1VvKN2O.png" width="33%"/>
	最佳報酬: 	
	
<img src="https://i.imgur.com/xFO8Q2m.png" width="40%"/>
	
## 實驗方法與步驟
本研究以Google Research Football環境(Kurach et al., 2019)模擬遊戲環境，選用USCG(Wei et al., 2017)[4] 實作online演算法，並嘗試分析該演算法效能以檢驗預期結果。因此，實驗步驟可區分為足球學習環境設定、演算法實作及演算法效能分析三大部分。
### 環境設置及介紹
利用Google Research Football 3D視覺化的環境(Kurach et al., 2019)模擬足球比賽。此步驟會運用該環境中提供的設定，調整為五對五雙隊伍模式，控制其中一個隊伍的所有球員，學習對抗另一個隊伍。
<img src="https://1.bp.blogspot.com/--ZLpcuDy8cg/XPqSR94pTsI/AAAAAAAAEMw/kx7V--J0oMMiA1wxjlCvNna9TISsafANgCLcBGAs/s1600/image9.gif" width="60%"/>  
環境狀態(state)為action執行後的環境狀態，包含球的位置、球員方位、pixel frame、分數、球員疲憊度等。玩家可採取動作空間(action)有二十種動作選擇，包含八種方向的移動及踢球等。
在報酬的計算上Kurach等人提供供兩種方式:score與checkpoint。由於遊戲在起初訓練時若只以score(進球才算分)計算報酬會很難給予學習方向，因此checkpoint多增加了其他判斷方式，如球在對手手中時，學習者越靠近球分數越高。
而環境的輸出則以observation表示，把觀察到的資料(raw observation) 轉化成下列任一種樣式作為input：
1. Pixels: 1280x720 RGB圖像，包含球員方位小圖及分數 
2. Super Mini Map: 4個72x96二元矩陣，表示球員與球的位置 
3. Floats: 115維度的矩陣，打包各種遊戲資訊。

### UCSG
UCSG可以有效降低regret，並保證regret有上界。該演算法無須調整學習率或exploration ratio，可自動達到探索與利用之間的平衡。演算法架構如下:

<img src="https://i.imgur.com/qO3uvZy.png" width="60%"/>

已知的輸入包含狀態空間S，兩個玩家的策略空間A，以及遊戲總步數T。演算法是以階段(phase)為單位，每單位都會進行四大步驟，分別為初始化找出轉移機率，利用轉移機率更新模型集合，再從模型區域中選擇最佳的策略與模型及最後一步執行所選擇之策略。由於是online的設定下，當每個階段達到該階段目標後，方可以進入至下一階段，因此每階段的執行長度也非固定，而是依過往觀測之數據而異。每階段的各個詳細步驟如下：
1. 依據該phase之前的觀測結果來初始化phase，並估計轉移機率 ${\hat{p}}_k(s'|s,a)$ 。
*   $v_k(s,a)$ 是phase $k$ 時(s,a)出現的次數，將它初始化為0。
*  $n_k(s,a)$ 是看之前的phase是否有出現過 $(s,a)$ 組合，如果沒看過的話將它設成1，若有則加總過去出現的次數
*  $n_k(s,a,s')$ 是看之前的phase出現過的結果來預測此phase的下一步，看過去經驗中 $(s,a)$的下一步是 $s'$ 的次數有多少次。
2. 更新信賴區間 $P_k(s,a)$ ，利用第一步結果更新每個轉移機率的信賴區間。 $CONF_1$ 的意義是希望將誤差的平均控制在某個範圍內， $CONF_2$ 的意義則是希望將誤差的標準控制在某個範圍內，透過信賴區間 $P_k(s,a)$ ，就可以建立出可能的stochastic game model，而真正的model $M$會高機率被包含在其中。
3. 選擇最佳model $M_k^1$ 和最佳策略 $\pi_k^1$ 。其中選擇最佳model的方法是從所有可能的model( $Mk$ 集合)利用 $Maximin-EVI$ 選出一個 $M_k^1$ 和 $\pi_k^1$ ， $Maximin-EVI$ 的方式就是估計選出的model和策略與最佳的model和策略之間要小於一個誤差 $\gamma_k$ 。也就是符合以下式子。
 $\min_{\pi^2}{\rho}\left(M_k^1,\pi_k^1,\pi^2,s\right)\geq\max_{\widetilde{M}\in\mathcal{M}_\mathcal{k}}{\rho^\ast}\left(\widetilde{M},s\right)-\gamma_k$ 
4. 執行策略。學習者會執行 $\pi_k^1$ ，並會不斷的去選出這個策略裡面的所有action並觀察他的reward以及下一個 $t+1$ 時間的狀態，由於已經觀察過了，所以把相對應的 $v_k(s,a)$ 次數加1，一直到某個 $(s,a)$ 在這個phase中出現的次數變成2倍為止。演算法中的迴圈總共會執行2n次，因為 $v_k(s,a)$ 一開始初始化為0，且在phase k-1時已經看過n次，phase k也要執行n次才能跳出迴圈，因此總共會變成 $n+n=2n$ 次。


## 實驗過程
由於硬體的限制，我們對Google Research Football的環境離散化，畫面中的資訊包含球員以及球的位置，以座標的方式儲存。並且只保留必要的action，將action從20個簡化到10個。將實驗設計從五人制足球改為Google Research Football中一對一的設定，即每隊只有一個球員，同時扮演進攻和防守的角色。  
此外，為了讓Google Research Football能夠符合我們的model，我們在reward設計的方面，以Google Research Football中checkpoint的設計為基礎，將能夠得到reward的位置拉大（進球給予1分的位置，其餘給予0.01分的reward）。  
在閱讀論文的過程中，我們在MAXIMIN-EVI這個演算法上面遇到了困難，因為論文的演算法中沒有提到應該如何找policy。最後我們的作法是讓 $\pi\left(s\right)$ 是一個退化的policy。並且針對每一個state s，先選一個能讓value最大的model，再從中挑出一個能讓value最大的action。  
我們使用了3個baseline來分析UCSG的效能，1. 是UCSG VS Nash Q-Learning，2. 是UCSG VS Random，3. 是UCSG VS DDQN。以reward與convergence作為主要評估基準，預期UCSG reward在相同步數下可以超過隨機及傳統MDP的Q-nash演算法與random policy。  

### 參數設置
在我們的環境中，UCSG、Nash Q-Learning、Random訓練過程使用的參數設置為：Trajectory（ $T$ ）= 100000，confidence parameter（ $\delta$ ）= 0.5，discount factor（ $\gamma$ ）=0.01，learning rate（ $\alpha$ ）= 0.9。DDQN的參數設置為：Trajectory（ $T$ ）= 100000， discount factor（ $\gamma$ ）=0.05，learning rate（ $\alpha$ ）= 0.01，Network Update = 800 steps。

## 實驗結果
### 收斂情況

<img src="https://i.imgur.com/i8GvFp4.png" width="70%"/>
圖表中的x軸代表步數，共10萬步，y軸代表我方agent獲得的收益。大約在兩萬步後，收益增加的幅度都一樣，代表UCSG會收斂，且可以看到UCSG的收斂有很明顯的規律。  
其他三者收斂情況則不太理想，Nash-Q的收斂圖沒有特別穩定的軌跡，在100,000步內未收斂成功。DDQN 每個episode有相似的頻率趨勢，然而reward都是遞減，只是loss 在測試200,000步之後，每episode最大的loss依然維持在-4左右無法減少。  

### reward差異

<img src="https://i.imgur.com/W7HQpvp.png" width="70%"/>

這是4個演算法的收益差異，可以看到大約在兩萬步後，收益增加的幅度都一樣，代表UCSG會收斂。而Nash-Q 的收益差異雖然越來越少，但也越來越接近 0。DDQN的收益差異大部分集中在1~-1之間，觀察情況為當我方持球時，多數可以取得較好的收益，然當對方持球時，無法阻擋對手的進攻趨勢。

### Total Reward

<img src="https://i.imgur.com/6Ah8iE8.png" width="70%"/>

UCSG大約在兩萬步後，reward增加的幅度都一樣，同樣代表著UCSG會收斂。  
Nash-Q 的total reward 逐漸遞增，但是整體而言不比UCSG好。DDQN total reward 逐漸遞減，且遞減的幅度看不出明顯變化。另外也可以觀察到UCSG在相同步數下，獲得的reward遠高於其他三者，且沒有負reward的情況，可以發現UCSG學習效率高於其他三者。

### 綜合比較

<img src="https://i.imgur.com/vwu9UUf.png" width="70%"/>

## 實驗限制及未來展望
### 實驗限制
我們原本想要實作5對5的足球比賽，但由於要控制的agent較多，要考量的state和action也更多，電腦在執行過程會當機，因此我們先簡化成1對1的足球比賽，觀察UCSG的實作成果。
### 未來展望
我們希望可以實作5對5足球賽，並且觀察agent的action是否會跟現實有所差異。另外特別想觀察的是，在真實生活的足球賽中，有些球員會做出「戰術性犯規」，也就是指後衛對進攻球員做出犯規行為，以阻止對手進攻，或者是己方極有可能被對手攻入一球的情況下故意手球犯規。雖然裁判會判給對手任意球或點球，但進球機會明顯比不犯規時的機會小。所以此時防守球員會對進攻球員進行戰術犯規或故意手球。未來想觀察agent是否能夠學習到這種行為。此外，由於UCSG的運算量非常大，前面有提到為了提升速度我們將環境做了離散化，我們希望未來也可以用其他方式提升運算速度，並且觀察其和現實球賽的差異。

## 參考文獻
[1] Kurach, Karol, et al. "Google research football: A novel reinforcement learning environment." arXiv preprint arXiv:1907.11180 (2019). 

[2] Wei, Chen-Yu, Yi-Te Hong, and Chi-Jen Lu. "Online reinforcement learning in stochastic games." Advances in Neural Information Processing Systems 30 (2017).

[3] Littman, Michael L. "Markov games as a framework for multi-agent reinforcement learning." Machine learning proceedings 1994. Morgan Kaufmann, 1994. 157-163.

[4] Brafman, Ronen I., and Moshe Tennenholtz. "R-max-a general polynomial time algorithm for near-optimal reinforcement learning." Journal of Machine Learning Research 3.Oct (2002): 213-231.

[5] Kearns, Michael, and Satinder Singh. "Near-optimal reinforcement learning in polynomial time." Machine learning 49.2 (2002): 209-232.

