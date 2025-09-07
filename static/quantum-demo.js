// Quantum Kernel Demo JavaScript

class QuantumDemo {
    constructor() {
        this.isLoading = false;
        this.init();
    }

    init() {
        // Check if running in iframe and adjust styling
        if (window.parent !== window) {
            document.body.classList.add('iframe-mode');
        }

        // Bind event listeners
        this.bindEvents();
        
        // Initialize tooltips
        this.initTooltips();
        
        // Add some initial quantum flair
        this.addQuantumEffects();
    }

    bindEvents() {
        const runButton = document.getElementById('run-button');
        if (runButton) {
            runButton.addEventListener('click', () => this.runSimulation());
        }

        // Add enter key support for inputs
        const inputs = document.querySelectorAll('input, select');
        inputs.forEach(input => {
            input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !this.isLoading) {
                    this.runSimulation();
                }
            });
        });

        // Add change listeners for real-time validation
        const depthInput = document.getElementById('depth');
        if (depthInput) {
            depthInput.addEventListener('change', this.validateDepth);
        }
    }

    validateDepth() {
        const depth = parseInt(this.value);
        const warningDiv = document.getElementById('depth-warning');
        
        // Remove existing warning
        if (warningDiv) {
            warningDiv.remove();
        }

        if (depth > 3) {
            const warning = document.createElement('div');
            warning.id = 'depth-warning';
            warning.className = 'warning-message';
            warning.innerHTML = '⚠️ Higher depths may take longer to compute';
            warning.style.cssText = `
                color: #ff6b35;
                font-size: 0.9em;
                margin-top: 5px;
                font-weight: 500;
            `;
            this.parentNode.appendChild(warning);
        }
    }

    initTooltips() {
        const tooltips = [
            {
                selector: '#feature_map',
                text: 'ZZ maps create entanglement between adjacent qubits, while Pauli maps use X, Y, Z rotations'
            },
            {
                selector: '#depth',
                text: 'Higher depth creates more complex quantum circuits but increases computation time'
            },
            {
                selector: '#ent',
                text: 'Full entanglement connects all qubits, linear only connects adjacent ones'
            }
        ];

        tooltips.forEach(({selector, text}) => {
            const element = document.querySelector(selector);
            if (element) {
                element.title = text;
                element.addEventListener('mouseenter', (e) => this.showTooltip(e, text));
                element.addEventListener('mouseleave', () => this.hideTooltip());
            }
        });
    }

    showTooltip(event, text) {
        // Simple tooltip implementation
        const tooltip = document.createElement('div');
        tooltip.className = 'custom-tooltip';
        tooltip.textContent = text;
        tooltip.style.cssText = `
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.8em;
            max-width: 250px;
            z-index: 1000;
            pointer-events: none;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        `;
        
        document.body.appendChild(tooltip);
        
        // Position tooltip
        const rect = event.target.getBoundingClientRect();
        tooltip.style.left = (rect.left + rect.width / 2 - tooltip.offsetWidth / 2) + 'px';
        tooltip.style.top = (rect.bottom + 10) + 'px';
    }

    hideTooltip() {
        const tooltip = document.querySelector('.custom-tooltip');
        if (tooltip) {
            tooltip.remove();
        }
    }

    addQuantumEffects() {
        // Add quantum accent to the run button
        const runButton = document.getElementById('run-button');
        if (runButton) {
            runButton.classList.add('quantum-accent');
        }

        // Add subtle animations to form elements
        const formGroups = document.querySelectorAll('.form-group');
        formGroups.forEach((group, index) => {
            group.style.animation = `fadeInUp 0.6s ease ${index * 0.1}s both`;
        });

        // Add CSS for fadeInUp animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
        `;
        document.head.appendChild(style);
    }

    showLoading() {
        this.isLoading = true;
        
        // Update button
        const runButton = document.getElementById('run-button');
        const buttonText = runButton.querySelector('.button-text');
        
        runButton.disabled = true;
        runButton.classList.add('loading');
        buttonText.textContent = 'Computing...';
        
        // Show loading container
        const loadingContainer = document.getElementById('loading-container');
        loadingContainer.classList.add('show');
        
        // Hide previous results
        this.hideResults();
        
        // Add random loading messages
        this.animateLoadingText();
    }

    hideLoading() {
        this.isLoading = false;
        
        // Update button
        const runButton = document.getElementById('run-button');
        const buttonText = runButton.querySelector('.button-text');
        
        runButton.disabled = false;
        runButton.classList.remove('loading');
        buttonText.textContent = 'Run Simulation';
        
        // Hide loading container
        const loadingContainer = document.getElementById('loading-container');
        loadingContainer.classList.remove('show');
    }

    animateLoadingText() {
        const messages = [
            'Initializing quantum circuits...',
            'Computing fidelity kernels...',
            'Training quantum SVM...',
            'Generating decision boundary...',
            'Rendering quantum visualization...'
        ];
        
        const loadingText = document.getElementById('loading-text');
        let messageIndex = 0;
        
        const interval = setInterval(() => {
            if (!this.isLoading) {
                clearInterval(interval);
                return;
            }
            
            loadingText.style.opacity = '0';
            setTimeout(() => {
                if (this.isLoading) {
                    loadingText.textContent = messages[messageIndex];
                    loadingText.style.opacity = '1';
                    messageIndex = (messageIndex + 1) % messages.length;
                }
            }, 300);
        }, 2000);
    }

    showResults(data) {
        const accuracyDisplay = document.getElementById('accuracy-display');
        const plotImage = document.getElementById('plot-image');
        const resultsSection = document.getElementById('results-section');
        
        // Animate accuracy display
        accuracyDisplay.style.transform = 'scale(0)';
        accuracyDisplay.textContent = `Accuracy: ${(data.accuracy * 100).toFixed(1)}%`;
        
        setTimeout(() => {
            accuracyDisplay.style.transition = 'transform 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55)';
            accuracyDisplay.style.transform = 'scale(1)';
        }, 100);
        
        // Update plot with fade-in effect
        plotImage.style.opacity = '0';
        plotImage.onload = () => {
            plotImage.style.transition = 'opacity 0.5s ease';
            plotImage.style.opacity = '1';
        };
        plotImage.src = data.plot_url;
        
        // Show results section
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    hideResults() {
        const resultsSection = document.getElementById('results-section');
        const accuracyDisplay = document.getElementById('accuracy-display');
        const plotImage = document.getElementById('plot-image');
        
        accuracyDisplay.textContent = '';
        plotImage.src = '';
    }

    showError(message) {
        // Remove existing error messages
        const existingError = document.querySelector('.error-message');
        if (existingError) {
            existingError.remove();
        }
        
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.innerHTML = `❌ ${message}`;
        
        const container = document.querySelector('.container');
        const resultsSection = document.getElementById('results-section');
        container.insertBefore(errorDiv, resultsSection);
        
        // Auto-remove error after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.style.transition = 'opacity 0.5s ease';
                errorDiv.style.opacity = '0';
                setTimeout(() => errorDiv.remove(), 500);
            }
        }, 5000);
        
        errorDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    async runSimulation() {
        if (this.isLoading) return;

        // Get form values
        const feature_map = document.getElementById('feature_map').value;
        const depth = document.getElementById('depth').value;
        const ent = document.getElementById('ent').value;

        // Validation
        if (!feature_map || !depth || !ent) {
            this.showError('Please fill in all parameters');
            return;
        }

        if (parseInt(depth) < 1 || parseInt(depth) > 5) {
            this.showError('Depth must be between 1 and 5');
            return;
        }

        this.showLoading();

        try {
            const response = await fetch('/run_kernel', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    feature_map: feature_map,
                    depth: parseInt(depth),
                    ent: ent
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Server error occurred');
            }

            if (data.error) {
                throw new Error(data.error);
            }

            this.hideLoading();
            this.showResults(data);

        } catch (error) {
            this.hideLoading();
            console.error('Simulation error:', error);
            this.showError(error.message || 'An unexpected error occurred');
        }
    }
}

// Initialize the demo when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new QuantumDemo();
});

// Add some global utility functions
window.quantumUtils = {
    // Function to export results (could be useful)
    exportResults: () => {
        const plotImage = document.getElementById('plot-image');
        const accuracy = document.getElementById('accuracy-display').textContent;
        
        if (plotImage.src && accuracy) {
            const link = document.createElement('a');
            link.download = `quantum-kernel-result-${Date.now()}.png`;
            link.href = plotImage.src;
            link.click();
        }
    },
    
    // Function to reset the demo
    reset: () => {
        document.getElementById('feature_map').value = 'zz';
        document.getElementById('depth').value = '2';
        document.getElementById('ent').value = 'full';
        
        const resultsSection = document.getElementById('results-section');
        resultsSection.style.display = 'none';
        
        const errorMessage = document.querySelector('.error-message');
        if (errorMessage) errorMessage.remove();
    }
};