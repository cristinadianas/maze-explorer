# Maze Explorer 2D – Reinforcement Learning

Environment custom Gymnasium pentru navigație 2D fără obstacole.
Proiectul va compara 4 algoritmi RL:
- Value Iteration (tabular)
- Q-Learning (tabular)
- SARSA (tabular)
- Deep Q-Network (DQN) (deep RL)

Framework principal: PyTorch + Stable Baselines3

# Comenzi de rulat in terminal:
- pip install -r requirements.txt
- python -m experiments.run_random_agent


# Afisare
Consola afișează evoluția unui agent care se mișcă aleatoriu într-un labirint 2D fără obstacole. La fiecare pas, grila este reafișată, unde A reprezintă agentul, G reprezintă ținta, iar . celulele libere. Agentul se deplasează pas cu pas până ajunge la G, iar la finalul episodului se afișează Total reward, calculat ca 100 minus numărul de pași făcuți. Aceasta se repetă pentru fiecare episod, oferind o vizualizare simplă a mediului și verificând funcționalitatea environment-ului înainte de implementarea algoritmilor RL.


