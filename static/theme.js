// theme.js - Handles theme switching and persistence

const applyTheme = (theme) => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('agriTheme', theme);
    
    // Update theme selector if it exists
    const selector = document.getElementById('theme-selector');
    if (selector) {
        selector.value = theme;
    }
};

const initTheme = () => {
    const savedTheme = localStorage.getItem('agriTheme') || 'dark';
    applyTheme(savedTheme);
};

// Apply immediately to prevent flashing
initTheme();

// Event listener setup (to be called after DOM load)
document.addEventListener('DOMContentLoaded', () => {
    const selector = document.getElementById('theme-selector');
    if (selector) {
        selector.value = localStorage.getItem('agriTheme') || 'dark';
        selector.addEventListener('change', (e) => {
            applyTheme(e.target.value);
            
            // If Chart.js exists on the page, update its colors dynamically
            if (typeof myChart !== 'undefined' && myChart) {
                myChart.options.scales.x.ticks.color = '#94A3B8';
                myChart.options.scales.y.ticks.color = '#94A3B8';
                myChart.options.plugins.legend.labels.color = '#F8FAFC';
                myChart.update();
            }
        });
    }
});
