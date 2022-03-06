//
// This Stan program defines Causal Support.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//


data {
  int<lower=0> nEvents;  // The number of events, E
  int<lower=0> nObjects; // The number of objects, O
  matrix[nEvents,nObjects] allObjects; // The (E,O) matrix
  int<lower=0,upper=1> activations[nEvents];  // An integer array of length E
  int<lower=0> nCombinations;
  int<lower=0,upper=1> allCombinations[nCombinations, nObjects]; // 
}

transformed data {
  matrix[nCombinations, nObjects] allCombinationsMat;
  allCombinationsMat = to_matrix(allCombinations);
  
}

// The parameters accepted by the model.
parameters {
  real<lower=0,upper=1> causalPower; // Scalar for causal power (i.e., all the same)
  real<lower=0,upper=1> pBlicket; // Scalar probability of being a blicket
  //real<lower=0,upper=1> backgroundStrength; // Scaler for strength of background cause
}

transformed parameters {
  vector[nCombinations] lp; // Log probability
  vector[nCombinations] np; // normalized probability
  vector[nEvents] liveBlickets;
  real pAct;
  real backgroundStrength;
  backgroundStrength = 0.01;
  
  for(c in 1:nCombinations){
    lp[c] = bernoulli_lpmf(allCombinations[c] | rep_vector(pBlicket, nObjects)); 
    liveBlickets = allObjects * allCombinationsMat[c]';
    for(i in 1:nEvents){
     // Noisy-OR parameterisation, everything has the same causal power.
     pAct = (1.0-(1.0-backgroundStrength)*pow(1.0-causalPower, liveBlickets[i])); // include background cause in the product (prior towards zero) //pow(real x, real y) returns x raised to the power of y
     lp[c] = lp[c] + bernoulli_lpmf(activations[i] | pAct);
    }
  }
  // Normalise posterior
  for(c in 1:nCombinations){
    np[c] = exp(lp[c]) / sum(exp(lp)); // return normalized probabilities
  }
  
}

// The model to be estimated.
model {
 causalPower ~ uniform(0, 1);
 pBlicket ~ uniform(0,1);
 target += log_sum_exp(lp);

}

