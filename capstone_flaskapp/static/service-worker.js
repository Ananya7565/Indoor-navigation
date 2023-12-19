// Service Worker Installation
this.addEventListener('install', event => {
    event.waitUntil(
      caches.open('my-cache-name-v1').then(cache => {
        return cache.addAll([
          '/',
          
          '/public/click.png',
          '/public/favicon.ico',
          
          '/public/logo192.png',
          '/public/logo512.png',
          '/public/maskable_icon.png',
          '/public/robots.txt',
          '/static/cd.jpg',
          '/static/newnewORB.csv',
          '/static/shortest_path_image.png',
          '/templates/index.html'
          
          // Add more paths to assets you want to cache
        ]).then(() => {
          console.log('Resources successfully cached');
        }).catch((error) => {
          console.error('Failed to cache resources:', error);
        });
      })
    );
  });
  
  // Service Worker Activation
  this.addEventListener('activate', event => {
    event.waitUntil(
      caches.keys().then(cacheNames => {
        return Promise.all(
          cacheNames.filter(cacheName => {
            return cacheName.startsWith('my-cache-name-v') && cacheName !== 'my-cache-name-v1';
          }).map(cacheName => {
            return caches.delete(cacheName);
          })
        );
      })
    );
  });
  
  // Service Worker Fetch
  this.addEventListener('fetch', event => {
    event.respondWith(
      caches.match(event.request).then((cachedResponse) => {
        return (cachedResponse) || fetch(event.request);
      })
    );
  });
  