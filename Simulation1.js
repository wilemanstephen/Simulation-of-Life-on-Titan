document.addEventListener("DOMContentLoaded", function () {
    const canvas = document.getElementById('envCanvas');
    const ctx = canvas.getContext('2d');
    const infoBox = document.getElementById('info');
    const SIZE = 20;
    const MOVE_PENALTY = 1;
    const MOUNTAIN_REWARD = 300;
    const EPSILON_DECAY = 0.9998;
    const LEARNING_RATE = 0.1;
    const DISCOUNT = 0.95;
    const SCALE = canvas.width / SIZE;
    let epsilon = 0.9;
    let qTable = {};
    let agent = new Agent();
    let resetCount = 0;
    let mountainTouches = 0;
    let simulationActive = true;

    function Agent() {
        this.x = 0;
        this.y = SIZE - 1;
    }

    Agent.prototype.action = function(choice) {
        if (choice === 0) this.x = Math.max(this.x - 1, 0);
        else if (choice === 1) this.x = Math.min(this.x + 1, SIZE - 1);
        else if (choice === 2) this.y = Math.max(this.y - 1, 0);
        else if (choice === 3) this.y = Math.min(this.y + 1, SIZE - 1);
    };

    function createMountain() {
        let mountain = Array.from({length: SIZE}, () => Array(SIZE).fill(0));
        const peakHeight = Math.floor(SIZE / 4);
        const baseWidth = Math.floor(SIZE / 4);
        const baseStartCol = SIZE - baseWidth;
        for (let i = 0; i < peakHeight; i++) {
            for (let j = baseStartCol + (peakHeight - i - 1); j < SIZE - (peakHeight - i - 1); j++) {
                mountain[i][j] = 1;
            }
        }
        return mountain;
    }

    let mountainRender = createMountain();

    function draw() {
        ctx.fillStyle = '#808080';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        for (let y = 0; y < SIZE; y++) {
            for (let x = 0; x < SIZE; x++) {
                ctx.fillStyle = mountainRender[y][x] === 1 ? 'red' : '#808080';
                ctx.fillRect(x * SCALE, y * SCALE, SCALE, SCALE);
            }
        }
        ctx.beginPath();
        ctx.arc(agent.x * SCALE + SCALE / 2, agent.y * SCALE + SCALE / 2, SCALE / 2, 0, 2 * Math.PI);
        ctx.fillStyle = 'green';
        ctx.fill();
        infoBox.textContent = `Resets: ${resetCount}, Mountain Touches: ${mountainTouches}, Agent Position: (${agent.x}, ${agent.y}), Epsilon: ${epsilon.toFixed(4)}`;
    }

    function simulate() {
        if (!simulationActive) return;
        let action = Math.floor(Math.random() * 4);
        agent.action(action);
        let collision = mountainRender[agent.y][agent.x] === 1;
        if (collision) {
            mountainTouches++;
            simulationActive = false;
        }
        let reward = collision ? MOUNTAIN_REWARD : -MOVE_PENALTY;
        let obs = `${agent.x},${agent.y}`;
        if (!qTable[obs]) qTable[obs] = Array(4).fill().map(() => Math.random() * -5);
        let maxFutureQ = Math.max(...qTable[obs]);
        let currentQ = qTable[obs][action];
        qTable[obs][action] = (1 - LEARNING_RATE) * currentQ + LEARNING_RATE * (reward + DISCOUNT * maxFutureQ);
        epsilon *= EPSILON_DECAY;
        draw();
        setTimeout(simulate, 100);
    }

    window.resetSimulation = function() {
        simulationActive = true;
        resetCount++;
        agent = new Agent();
        simulate();
    };

    window.clearMemory = function() {
        epsilon = 0.9;
        qTable = {};
        mountainTouches = 0;
        resetCount = 0;
        simulationActive = true;
        simulate();
    };

    simulate();
});