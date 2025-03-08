# CCS_E2CO-RL

Code for 
["Optimization of pressure management strategies for geological CO2 storage using surrogate model-based reinforcement learning"]((https://www.sciencedirect.com/science/article/pii/S1750583624002056)). (under construction)



step 1: run MSE2C_Ksteps.ipynb to construct E2CO proxy models (either one should work, maybe MSE2C_Ksteps_SC.ipynb work best)


step 2: run RL_SAC_train.ipynb to employ RL to optimize

## Abstract
Injecting greenhouse gas (e.g. CO2) into deep underground reservoirs for permanent storage can inadvertently lead to fault reactivation, caprock fracturing and greenhouse gas leakage when the injection-induced stress exceeds the critical threshold. It is essential to monitor the evolution of pressure and the movement of the CO2 plume closely during the injection to allow for timely remediation actions or rapid adjustments to the storage design. Extraction of pre-existing fluids at various stages of the injection process, referred as pressure management, can mitigate associated risks and lessen environmental impact. However, identifying optimal pressure management strategies typically requires thousands of simulations, making the process computationally prohibitive. This paper introduces a novel surrogate model-based reinforcement learning method for devising optimal pressure management strategies for geological CO2 sequestration efficiently. Our approach comprises of two steps. The first step involves developing a surrogate model using the embed to control method, which employs an encoder-transition-decoder structure to learn dynamics in a latent or reduced space. The second step, leveraging this proxy model, utilizes reinforcement learning to find an optimal strategy that maximizes economic benefits while satisfying various control constraints. The reinforcement learning agent receives the latent state representation and immediate reward tailored for CO2 sequestration and choose real-time controls which are subject to predefined engineering constraints in order to maximize the long-term cumulative rewards. To demonstrate its effectiveness, this framework is applied to a compositional simulation model where CO2 is injected into saline aquifer. The results reveal that our surrogate model-based reinforcement learning approach significantly optimizes CO2 sequestration strategies, leading to notable economic gains compared to baseline scenarios.


## Workflow

![alt text](workflow)

## Citation
If you find our research helpful, please consider citing us with：

```
@article{chen2024optimization,
  title={Optimization of pressure management strategies for geological CO2 storage using surrogate model-based reinforcement learning},
  author={Chen, Jungang and Gildin, Eduardo and Kompantsev, Georgy},
  journal={International Journal of Greenhouse Gas Control},
  volume={138},
  pages={104262},
  year={2024},
  publisher={Elsevier}
}
```
