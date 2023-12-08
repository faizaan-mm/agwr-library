<template>
  <div class="container">
    <h1 class="title">Predict Mortality in Tokyo</h1>
    <br>
    <form @submit.prevent="predictMortality" class="form">
      <label for="eb2564" class="label">Employment Rate (25-64):</label>
      <input
        type="number"
        id="eb2564"
        v-model="eb2564"
        required
        class="input"
        step="0.00001"
      />
      <br />
      <label for="occTec" class="label">% in Technology Occupation:</label>
      <input
        type="number"
        id="occTec"
        v-model="occTec"
        required
        class="input"
        step="0.00001"
      />
      <br />
      <label for="ownh" class="label">% in Owner-Occupied Housing:</label>
      <input
        type="number"
        id="ownh"
        v-model="ownh"
        required
        class="input"
        step="0.00001"
      />
      <br />
      <label for="pop65" class="label">Population over 65:</label>
      <input
        type="number"
        id="pop65"
        v-model="pop65"
        required
        class="input"
        step="0.00001"
      />
      <br />
      <label for="unemp" class="label">Unemployment Rate:</label>
      <input
        type="number"
        id="unemp"
        v-model="unemp"
        required
        class="input"
        step="0.00001"
      />
      <br />
      <label for="x_cent" class="label">X Centroid:</label>
      <input
        type="number"
        id="unemp"
        v-model="x_cent"
        required
        class="input"
        step="0.00001"
      />
      <br />
      <label for="y_cent" class="label">Y Centroid:</label>
      <input
        type="number"
        id="unemp"
        v-model="y_cent"
        required
        class="input"
        step="0.00001"
      />
      <br />
      <button type="submit" class="button" @click="predictMortality">Predict</button>
    </form>
    <div v-if="prediction" class="prediction-container">
      <p class="prediction-label">Predicted Mortality Rate:</p>
      <span class="prediction-value">{{ prediction }}</span>
    </div>
  </div>
</template>

<style scoped>
.container {
  font-family: sans-serif;
  margin: 0 auto;
  width: 400px;
  padding: 20px;
  border: 1px solid #ddd;
  border-radius: 5px;
}

.title {
  font-weight: bold;
  text-align: center;
  margin-bottom: 10px;
}

.form {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.label {
  font-weight: bold;
  margin-bottom: 5px;
}

.input {
  padding: 5px 10px;
  border: 1px solid #ccc;
  border-radius: 3px;
  outline: none;
}

.button {
  background-color: #4CAF50;
  color: white;
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.prediction-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 20px;
}

.prediction-label {
  font-weight: bold;
  margin-bottom: 5px;
}

.prediction-value {
  font-weight: bold;
  color: #4CAF50;
}
</style>

<script>
import axios from 'axios';
export default {
  data() {
    return {
      prediction: null,
    };
  },
  methods: {
  predictMortality() {

    axios.post('http://127.0.0.1:8888/predict/', {
      headers: {
        'Content-Type': 'application/json'
      },
      data: JSON.stringify({
    "model": "trained_model_pysalTokyo-SMGWR-RF.pickle",
    "x_test": [
      this.eb2564,
      this.occTec,
      this.ownh,
      this.pop65,
      this.unemp
    ],
    "coords_test": [
      this.x_cent,
      this.y_cent
    ]
  })
    })
    .then((response) => {
      // Handle successful response
      alert(response.data.data);
      console.log(response.data.data);
    })
    .catch((error) => {
      // Handle error
      console.error(error);
    });

  },
},

};
</script>
