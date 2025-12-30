// Corn Leaf Disease Classifier - Interactive Enhancements

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    
    // Add smooth scrolling behavior
    document.documentElement.style.scrollBehavior = 'smooth';
    
    // Add loading animation for images
    function addImageLoadingEffect() {
        const images = document.querySelectorAll('img');
        images.forEach(img => {
            img.addEventListener('load', function() {
                this.style.opacity = '0';
                this.style.transform = 'scale(0.8)';
                this.style.transition = 'all 0.5s ease';
                
                setTimeout(() => {
                    this.style.opacity = '1';
                    this.style.transform = 'scale(1)';
                }, 100);
            });
        });
    }
    
    // Add hover effects to cards
    function addCardHoverEffects() {
        const cards = document.querySelectorAll('.result-card, .info-card');
        cards.forEach(card => {
            card.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-5px) scale(1.02)';
                this.style.boxShadow = '0 15px 40px rgba(0, 0, 0, 0.15)';
            });
            
            card.addEventListener('mouseleave', function() {
                this.style.transform = 'translateY(0) scale(1)';
                this.style.boxShadow = '0 6px 25px rgba(0, 0, 0, 0.08)';
            });
        });
    }
    
    // Add progress bar animation
    function animateProgressBars() {
        const progressBars = document.querySelectorAll('.prob-bar > div');
        progressBars.forEach((bar, index) => {
            const width = bar.style.width;
            bar.style.width = '0%';
            
            setTimeout(() => {
                bar.style.width = width;
            }, index * 200 + 500);
        });
    }
    
    // Add typing effect to main title
    function addTypingEffect() {
        const title = document.querySelector('.main-header h1');
        if (title) {
            const text = title.textContent;
            title.textContent = '';
            title.style.borderRight = '2px solid white';
            
            let i = 0;
            const typeWriter = () => {
                if (i < text.length) {
                    title.textContent += text.charAt(i);
                    i++;
                    setTimeout(typeWriter, 100);
                } else {
                    setTimeout(() => {
                        title.style.borderRight = 'none';
                    }, 1000);
                }
            };
            
            setTimeout(typeWriter, 500);
        }
    }
    
    // Add floating animation to upload section
    function addFloatingAnimation() {
        const uploadSection = document.querySelector('.upload-section');
        if (uploadSection) {
            let isFloating = false;
            
            const float = () => {
                if (!isFloating) {
                    isFloating = true;
                    uploadSection.style.transform = 'translateY(-10px)';
                    setTimeout(() => {
                        uploadSection.style.transform = 'translateY(0px)';
                        isFloating = false;
                    }, 2000);
                }
            };
            
            setInterval(float, 4000);
        }
    }
    
    // Add particle effect to header
    function addParticleEffect() {
        const header = document.querySelector('.main-header');
        if (header) {
            for (let i = 0; i < 20; i++) {
                const particle = document.createElement('div');
                particle.style.cssText = `
                    position: absolute;
                    width: 4px;
                    height: 4px;
                    background: rgba(255, 255, 255, 0.6);
                    border-radius: 50%;
                    pointer-events: none;
                    animation: float ${3 + Math.random() * 4}s infinite ease-in-out;
                    left: ${Math.random() * 100}%;
                    top: ${Math.random() * 100}%;
                    animation-delay: ${Math.random() * 2}s;
                `;
                header.appendChild(particle);
            }
        }
    }
    
    // Add CSS for particle animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); opacity: 0.6; }
            50% { transform: translateY(-20px) rotate(180deg); opacity: 1; }
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .shake {
            animation: shake 0.5s ease-in-out;
        }
        
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-5px); }
            75% { transform: translateX(5px); }
        }
    `;
    document.head.appendChild(style);
    
    // Add intersection observer for animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe elements for scroll animations
    const animateOnScroll = document.querySelectorAll('.result-card, .prediction-box, .prob-container');
    animateOnScroll.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'all 0.6s ease';
        observer.observe(el);
    });
    
    // Add click effect to buttons
    function addButtonClickEffect() {
        const buttons = document.querySelectorAll('button');
        buttons.forEach(button => {
            button.addEventListener('click', function(e) {
                const ripple = document.createElement('span');
                const rect = this.getBoundingClientRect();
                const size = Math.max(rect.width, rect.height);
                const x = e.clientX - rect.left - size / 2;
                const y = e.clientY - rect.top - size / 2;
                
                ripple.style.cssText = `
                    position: absolute;
                    width: ${size}px;
                    height: ${size}px;
                    left: ${x}px;
                    top: ${y}px;
                    background: rgba(255, 255, 255, 0.6);
                    border-radius: 50%;
                    transform: scale(0);
                    animation: ripple 0.6s linear;
                    pointer-events: none;
                `;
                
                this.style.position = 'relative';
                this.style.overflow = 'hidden';
                this.appendChild(ripple);
                
                setTimeout(() => {
                    ripple.remove();
                }, 600);
            });
        });
    }
    
    // Add ripple animation CSS
    const rippleStyle = document.createElement('style');
    rippleStyle.textContent = `
        @keyframes ripple {
            to {
                transform: scale(4);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(rippleStyle);
    
    // Initialize all effects
    setTimeout(() => {
        addImageLoadingEffect();
        addCardHoverEffects();
        addFloatingAnimation();
        addParticleEffect();
        addButtonClickEffect();
        
        // Check for progress bars and animate them
        const checkForProgressBars = setInterval(() => {
            const progressBars = document.querySelectorAll('.prob-bar > div');
            if (progressBars.length > 0) {
                animateProgressBars();
                clearInterval(checkForProgressBars);
            }
        }, 1000);
        
    }, 1000);
    
    // Add smooth transitions for Streamlit rerun
    const observer2 = new MutationObserver(() => {
        setTimeout(() => {
            addImageLoadingEffect();
            addCardHoverEffects();
            addButtonClickEffect();
        }, 500);
    });
    
    observer2.observe(document.body, {
        childList: true,
        subtree: true
    });
});

// Add performance monitoring
window.addEventListener('load', () => {
    console.log('🌽 Corn Leaf Disease Classifier loaded successfully!');
    
    // Add loading complete animation
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 0.5s ease';
    
    setTimeout(() => {
        document.body.style.opacity = '1';
    }, 100);
});