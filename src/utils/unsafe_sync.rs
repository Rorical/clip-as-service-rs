use std::ops::{Deref, DerefMut};

#[derive(Debug)]
pub struct UnsafeSync<T>(pub T);
unsafe impl<T> Sync for UnsafeSync<T> {}

impl<T> UnsafeSync<T> {
    pub fn new(internal: T) -> Self { Self(internal) }
}

impl<T> AsMut<T> for UnsafeSync<T> {
    fn as_mut(&mut self) -> &mut T { &mut self.0 }
}

impl<T> AsRef<T> for UnsafeSync<T> {
    fn as_ref(&self) -> &T { &self.0 }
}

impl<T> Clone for UnsafeSync<T>
    where
        T: Clone,
{
    fn clone(&self) -> Self { Self::new(self.0.clone()) }
}


impl<T> Default for UnsafeSync<T>
    where
        T: Default,
{
    fn default() -> Self { Self(T::default()) }
}

impl<T> Deref for UnsafeSync<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<T> DerefMut for UnsafeSync<T> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl<T> From<T> for UnsafeSync<T> {
    fn from(other: T) -> Self { Self(other) }
}


/// A wrapper type that is always `Send` and `Sync`.
pub struct UnsafeSendSync<T>(pub T);
unsafe impl<T> Send for UnsafeSendSync<T> {}
unsafe impl<T> Sync for UnsafeSendSync<T> {}

impl<T> UnsafeSendSync<T> {
    pub fn new(internal: T) -> Self { Self(internal) }
}

impl<T> AsMut<T> for UnsafeSendSync<T> {
    fn as_mut(&mut self) -> &mut T { &mut self.0 }
}

impl<T> AsRef<T> for UnsafeSendSync<T> {
    fn as_ref(&self) -> &T { &self.0 }
}

impl<T> Default for UnsafeSendSync<T>
    where
        T: Default,
{
    fn default() -> Self { Self(T::default()) }
}

impl<T> Deref for UnsafeSendSync<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<T> DerefMut for UnsafeSendSync<T> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl<T> Clone for UnsafeSendSync<T>
    where
        T: Clone,
{
    fn clone(&self) -> Self { Self::new(self.0.clone()) }
}

impl<T> From<T> for UnsafeSendSync<T> {
    fn from(other: T) -> Self { Self(other) }
}